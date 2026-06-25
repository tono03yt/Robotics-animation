#include <pthread.h>
/*
* ==============================================================================
* IPC Receiver & Robot Controller (With Retry Logic)
* ==============================================================================
*/
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <signal.h>
#include <errno.h>
#include <fcntl.h>
#include <termios.h>

#define DEFAULT_SOCK_PATH "/tmp/robot_pipeline.sock"
#define BACKLOG 8
#define BUF_SIZE 4096

static int g_server_fd = -1;
static const char *g_sock_path = NULL;
static int arduino_fd = -1;
static char saved_port[256] = {0};

/* Proportional control parameters */
static float pan_angle = 90.0f;
static float tilt_angle = 90.0f;

static float pan_integral = 0.0f;
static float tilt_integral = 0.0f;
static float last_x_error = 0.0f;
static float last_y_error = 0.0f;

static struct timespec last_ts = {0};

#define PAN_CENTER 90.0f
#define PAN_MIN 60.0f
#define PAN_MAX 120.0f

#define TILT_CENTER 90.0f
#define TILT_MIN 75.0f
#define TILT_MAX 105.0f

#define KP_X 15.0f
#define KI_X 0f
#define KD_X 0.1f


#define KP_Y 10.0f
#define KI_Y 0f
#define KD_Y 0.1f

#define INTEGRAL_LIMIT_X 20.0f
#define INTEGRAL_LIMIT_Y 20.0f

#define deadzone_x 0.1f
#define deadzone_y 0.1f

#define direction_x -1.0f
#define direction_y 1.0f

static void cleanup(void) {
    if (arduino_fd >= 0) {
        char cmd[32];
        snprintf(cmd, sizeof(cmd), "%d,%d\n", (int)PAN_CENTER, (int)TILT_CENTER);
        write(arduino_fd, cmd, strlen(cmd));
        close(arduino_fd);
        arduino_fd = -1;
    }
    if (g_server_fd >= 0) {
        close(g_server_fd);
        g_server_fd = -1;
    }
    if (g_sock_path) {
        unlink(g_sock_path);
        printf("\n[IPC] Socket %s removed.\n", g_sock_path);
    }
}

static void on_signal(int sig) {
    (void)sig;
    cleanup();
    exit(0);
}

static int setup_serial(const char *port) {
    int fd = open(port, O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd == -1) {
        return -1;
    }

    struct termios options;
    tcgetattr(fd, &options);

    cfsetispeed(&options, B115200);
    cfsetospeed(&options, B115200);

    options.c_cflag &= ~PARENB;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;
    options.c_cflag &= ~CRTSCTS;
    options.c_cflag |= CREAD | CLOCAL;

    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_oflag &= ~OPOST;

    tcsetattr(fd, TCSANOW, &options);
    fcntl(fd, F_SETFL, FNDELAY); 

    printf("[Serial] Connected to %s at 115200 baud.\n", port);
    sleep(2);
    return fd;
}

static void ensure_serial_connection() {
    if (saved_port[0] == '\0') return; // User chose to skip serial

    if (arduino_fd < 0) {
        arduino_fd = setup_serial(saved_port);
        if (arduino_fd < 0) {
            printf("[Serial] Could not open %s. Retrying in background...\n", saved_port);
        }
    }
}

static float clampf(float v, float min, float max) {
    if (v < min) return min;
    if (v > max) return max;
    return v;
}

static float get_dt_seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);

    if (last_ts.tv_sec == 0 && last_ts.tv_nsec == 0) {
        last_ts = now;
        return 0.02f;
    }

    float dt = (float)(now.tv_sec - last_ts.tv_sec) +
               (float)(now.tv_nsec - last_ts.tv_nsec) / 1000000000.0f;

    last_ts = now;

    if (dt < 0.001f) dt = 0.001f;
    if (dt > 0.1f)   dt = 0.1f;
    return dt;
}


#define DEADBAND 0.12f
#define PI_LIMIT 0.27f
#define PD_LIMIT 0.33f

#define KP_PAN  2.5f
#define KI_PAN  0.03f
#define KD_PAN  2.5f
#define K_PD_PAN 3.5f

#define KP_TILT 2.5f
#define KI_TILT 0.03f
#define KD_TILT 2.5f
#define K_PD_TILT 2.8f

static void handle_pos(float x, float y, float conf) {
    printf("[POS] x=%-8.4f y=%-8.4f conf=%.2f\n", x, y, conf);

    // float dt = get_dt_seconds(); // Not strictly used for math to match Arduino 1:1, but keep for timing if needed
    get_dt_seconds(); // call to keep timestamp updated

    if (conf < 0.5f) {
        pan_integral = 0.0f;
        tilt_integral = 0.0f;
        last_x_error = 0.0f;
        last_y_error = 0.0f;

        if (pan_angle > PAN_CENTER) pan_angle -= 0.5f;
        else if (pan_angle < PAN_CENTER) pan_angle += 0.5f;

        if (tilt_angle > TILT_CENTER) tilt_angle -= 0.5f;
        else if (tilt_angle < TILT_CENTER) tilt_angle += 0.5f;
    } else {
        float panError = x;
        float tiltError = y;

        // Totzone
        if (panError > -DEADBAND && panError < DEADBAND) panError = 0.0f;
        if (tiltError > -DEADBAND && tiltError < DEADBAND) tiltError = 0.0f;

        float panDerivative = panError - last_x_error;
        float tiltDerivative = tiltError - last_y_error;

        float absPan = (panError > 0.0f) ? panError : -panError;
        float absTilt = (tiltError > 0.0f) ? tiltError : -tiltError;

        // Integrator nur im PI-Bereich aktiv
        if (absPan < PI_LIMIT) {
            pan_integral += panError;    
            pan_integral = clampf(pan_integral, -20.0f, 20.0f);
        }

        if (absTilt < PI_LIMIT) {
            tilt_integral += tiltError;
            tilt_integral = clampf(tilt_integral, -20.0f, 20.0f);
        }

        // PI-Regler
        float panPI = KP_PAN * panError + KI_PAN * pan_integral;
        float tiltPI = KP_TILT * tiltError + KI_TILT * tilt_integral;

        // PD-Regler
        float panPD = K_PD_PAN * panError + KD_PAN * panDerivative;
        float tiltPD = K_PD_TILT * tiltError + KD_TILT * tiltDerivative;

        // Überblendfaktor
        float panBlend;
        float tiltBlend;

        // PAN Blending
        if (absPan <= PI_LIMIT) panBlend = 0.0f;
        else if (absPan >= PD_LIMIT) panBlend = 1.0f;
        else panBlend = (absPan - PI_LIMIT) / 0.06f;

        // TILT Blending
        if (absTilt <= PI_LIMIT) tiltBlend = 0.0f;
        else if (absTilt >= PD_LIMIT) tiltBlend = 1.0f;
        else tiltBlend = (absTilt - PI_LIMIT) / 0.06f;

        // Gain Scheduling Output
        float panOutput = (1.0f - panBlend) * panPI + panBlend * panPD;
        float tiltOutput = (1.0f - tiltBlend) * tiltPI + tiltBlend * tiltPD;

        // In your arduino code: panPos -= panOutput; tiltPos -= tiltOutput;
        // In the original C script we used direction_x/y (1.0 and -1.0).
        // I am subtracting directly to match the exact mathematical intent from your Arduino snippet.
        // If the axes are inverted, change these to +=
        pan_angle -= panOutput;
        tilt_angle -= tiltOutput;           

        last_x_error = panError;
        last_y_error = tiltError;
    }

    // Begrenzen (Matching your Arduino ranges while using the clamp function)
    pan_angle = clampf(pan_angle, 20.0f, 160.0f);
    tilt_angle = clampf(tilt_angle, 45.0f, 135.0f);

    ensure_serial_connection();

    if (arduino_fd >= 0) {
        char cmd[32];
        snprintf(cmd, sizeof(cmd), "%d,%d\n", (int)pan_angle, (int)tilt_angle);
        if (write(arduino_fd, cmd, strlen(cmd)) < 0) {
            printf("[Serial] Connection lost. Attempting to reconnect to %s...\n", saved_port);
            close(arduino_fd);
            arduino_fd = -1;
        }
    }
}

static void handle_anim(const char *animation, const char *text) {
    printf("[ANIM] animation=%-10s text=%s\n", animation, text);

    ensure_serial_connection();
    if (arduino_fd >= 0) {
        char cmd[128];
        snprintf(cmd, sizeof(cmd), "anim,%s\n", animation);
        if (write(arduino_fd, cmd, strlen(cmd)) < 0) {
            printf("[Serial] Connection lost during animation. Attempting to reconnect to %s...\n", saved_port);
            close(arduino_fd);
            arduino_fd = -1;
        }
    }
}

static int json_str(const char *json, const char *key, char *out, int out_sz) {
    char search[64];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *p = strstr(json, search);
    if (!p) return 0;
    p += strlen(search);
    while (*p == ' ' || *p == ':') p++;
    if (*p == '"') {
        p++;
        int i = 0;
        while (*p && *p != '"' && i < out_sz - 1)
            out[i++] = *p++;
        out[i] = '\0';
        return 1;
    }
    return 0;
}

static int json_float(const char *json, const char *key, float *out) {
    char search[64];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *p = strstr(json, search);
    if (!p) return 0;
    p += strlen(search);
    while (*p == ' ' || *p == ':') p++;
    *out = strtof(p, NULL);
    return 1;
}

static void dispatch(const char *line) {
    char type[32] = {0};
    if (!json_str(line, "type", type, sizeof(type))) return;

    if (strcmp(type, "pos") == 0) {
        float x = 0, y = 0, conf = 0;
        json_float(line, "x", &x);
        json_float(line, "y", &y);
        json_float(line, "conf", &conf);
        handle_pos(x, y, conf);
    } else if (strcmp(type, "anim") == 0) {
        char animation[64] = {0};
        char text[512] = {0};
        json_str(line, "animation", animation, sizeof(animation));
        json_str(line, "text", text, sizeof(text));
        handle_anim(animation, text);
    } else {
        printf("[IPC] Unknown message type: %s\n", type);
    }
}

static void *client_thread(void *arg) {
    int client_fd = (int)(intptr_t)arg;
    char buf[BUF_SIZE];
    char line[BUF_SIZE];
    int line_len = 0;
    ssize_t n;

    while ((n = read(client_fd, buf, sizeof(buf) - 1)) > 0) {
        buf[n] = '\0';
        for (int i = 0; i < (int)n; i++) {
            if (buf[i] == '\n') {
                line[line_len] = '\0';
                if (line_len > 0)
                    dispatch(line);
                line_len = 0;
            } else if (line_len < (int)sizeof(line) - 1) {
                line[line_len++] = buf[i];
            }
        }
    }
    if (line_len > 0) {
        line[line_len] = '\0';
        dispatch(line);
    }
    close(client_fd);
    printf("[IPC] Python disconnected.\n");
    return NULL;
}

static const char *select_socket_path(int argc, char *argv[]) {
    if (argc >= 2) {
        return argv[1];
    }
    return DEFAULT_SOCK_PATH;
}

static void select_serial_interface(void) {
    printf("\n ╔══════════════════════════════════════════════╗\n");
    printf(" ║     Arduino Serial Interface Setup           ║\n");
    printf(" ╚══════════════════════════════════════════════╝\n\n");

    printf(" Available serial ports (from /dev/ttyACM* and /dev/ttyUSB*):\n");
    system("ls -1 /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | awk '{print \"   [\" NR \"] \" $1}'");

    printf("\n Enter port path (e.g., /dev/ttyACM0) or press Enter to skip:\n");
    printf(" > ");
    fflush(stdout);

    char input[256];
    if (!fgets(input, sizeof(input), stdin) || input[0] == '\n' || input[0] == '\0') {
        printf("[Serial] Skipped serial setup. Running without Arduino.\n");
        return;
    }

    input[strcspn(input, "\r\n")] = '\0';

    // Save the port path for background retries
    strncpy(saved_port, input, sizeof(saved_port) - 1);

    arduino_fd = setup_serial(saved_port);
    if (arduino_fd < 0) {
        printf("[Serial] Warning: Could not open %s right now. The script will retry in the background.\n", saved_port);
    }
}

int main(int argc, char *argv[]) {
    select_serial_interface();
    g_sock_path = select_socket_path(argc, argv);

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    g_server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (g_server_fd < 0) {
        perror("[IPC] socket()");
        return 1;
    }

    unlink(g_sock_path);

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, g_sock_path, sizeof(addr.sun_path) - 1);

    if (bind(g_server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("[IPC] bind()");
        close(g_server_fd);
        return 1;
    }

    if (listen(g_server_fd, BACKLOG) < 0) {
        perror("[IPC] listen()");
        cleanup();
        return 1;
    }

    printf("\n[IPC] ✓ Listening on: %s\n", g_sock_path);
    printf("[IPC] Start the Python backend now — it will auto-connect.\n");
    printf("[IPC] Press Ctrl+C to quit.\n\n");

    while (1) {
        int client_fd = accept(g_server_fd, NULL, NULL);
        if (client_fd < 0) {
            if (errno == EINTR) break;
            perror("[IPC] accept()");
            continue;
        }

        printf("[IPC] Python connected.\n");
        pthread_t t;
        pthread_create(&t, NULL, client_thread, (void *)(intptr_t)client_fd);
        pthread_detach(t);
    }

    cleanup();
    return 0;
}