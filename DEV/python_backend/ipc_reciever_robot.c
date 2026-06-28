/*
 * ==============================================================================
 * IPC Receiver & Robot Controller
 * PID Refactor + Soft Deadband + Float Servo Targets
 * ==============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <fcntl.h>
#include <termios.h>
#include <pthread.h>
#include <stdint.h>
#include <time.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/types.h>

/* ==============================================================================
 * Settings & Constants
 * ============================================================================== */

/* IPC Server Settings */
#define DEFAULT_SOCK_PATH "/tmp/robot_pipeline.sock"
#define BACKLOG 8
#define BUF_SIZE 4096

/* Servo Constraints */
#define PAN_CENTER   90.0f
#define PAN_MIN      20.0f
#define PAN_MAX      160.0f

#define TILT_CENTER  90.0f
#define TILT_MIN     45.0f
#define TILT_MAX     135.0f

/* PID Controller Settings */
#define DEADBAND             0.08f

#define KP_PAN              12.0f
#define KI_PAN               0.035f
#define KD_PAN               1.3f

#define KP_TILT             11.0f
#define KI_TILT              0.035f
#define KD_TILT              1.5f

#define PAN_INTEGRAL_MAX     2.0f
#define TILT_INTEGRAL_MAX    2.0f

#define PAN_STEP_MAX         3.0f
#define TILT_STEP_MAX        2.5f
#define CENTER_RETURN_ALPHA  0.08f

/* Rate limiting */
#define SERIAL_UPDATE_INTERVAL 0.02f   /* 50 Hz */

/* ==============================================================================
 * Global Variables
 * ============================================================================== */

static int g_server_fd = -1;
static const char *g_sock_path = NULL;
static int arduino_fd = -1;
static char saved_port[256] = {0};

static struct timespec last_send_ts = {0};
static struct timespec last_ts = {0};

static float pan_angle = PAN_CENTER;
static float tilt_angle = TILT_CENTER;
static float last_x_error = 0.0f;
static float last_y_error = 0.0f;
static float pan_integral = 0.0f;
static float tilt_integral = 0.0f;

/* ==============================================================================
 * System & Cleanup Functions
 * ============================================================================== */

static void cleanup(void) {
    if (arduino_fd >= 0) {
        char cmd[64];
        snprintf(cmd, sizeof(cmd), "%.2f,%.2f\n", PAN_CENTER, TILT_CENTER);
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

/* ==============================================================================
 * Serial Communication Functions
 * ============================================================================== */

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
    options.c_iflag = 0;

    tcsetattr(fd, TCSANOW, &options);
    fcntl(fd, F_SETFL, FNDELAY);

    printf("[Serial] Connected to %s at 115200 baud.\n", port);
    sleep(2);
    return fd;
}

static void ensure_serial_connection(void) {
    if (saved_port[0] == '\0') {
        return;
    }

    if (arduino_fd < 0) {
        arduino_fd = setup_serial(saved_port);
        if (arduino_fd < 0) {
            printf("[Serial] Could not open %s. Retrying in background...\n", saved_port);
        }
    }
}

/* ==============================================================================
 * Math & Timing Helpers
 * ============================================================================== */

static float clampf(float v, float min, float max) {
    if (v < min) return min;
    if (v > max) return max;
    return v;
}

static float apply_soft_deadband(float e, float band) {
    if (e > -band && e < band) return 0.0f;
    if (e > 0.0f) return e - band;
    return e + band;
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
    if (dt > 0.1f) dt = 0.1f;
    return dt;
}

static void send_servo_targets(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);

    float elapsed_send = (float)(now.tv_sec - last_send_ts.tv_sec) +
                         (float)(now.tv_nsec - last_send_ts.tv_nsec) / 1000000000.0f;

    if (elapsed_send < SERIAL_UPDATE_INTERVAL) {
        return;
    }

    last_send_ts = now;

    ensure_serial_connection();
    if (arduino_fd >= 0) {
        char cmd[64];
        snprintf(cmd, sizeof(cmd), "%.2f,%.2f\n", pan_angle, tilt_angle);
        if (write(arduino_fd, cmd, strlen(cmd)) < 0) {
            printf("[Serial] Connection lost. Attempting to reconnect to %s...\n", saved_port);
            close(arduino_fd);
            arduino_fd = -1;
        }
    }
}

/* ==============================================================================
 * Robot Control & Dispatch
 * ============================================================================== */

static void handle_pos(float x, float y, float conf) {
    printf("[POS] x=%-8.4f y=%-8.4f conf=%.2f\n", x, y, conf);

    float dt = get_dt_seconds();

    if (conf < 0.5f) {
        last_x_error = 0.0f;
        last_y_error = 0.0f;
        pan_integral = 0.0f;
        tilt_integral = 0.0f;

        pan_angle  += (PAN_CENTER  - pan_angle)  * CENTER_RETURN_ALPHA;
        tilt_angle += (TILT_CENTER - tilt_angle) * CENTER_RETURN_ALPHA;
    } else {
        float pan_error = apply_soft_deadband(x, DEADBAND);
        float tilt_error = apply_soft_deadband(y, DEADBAND);

        pan_integral += pan_error * dt;
        tilt_integral += tilt_error * dt;

        pan_integral = clampf(pan_integral, -PAN_INTEGRAL_MAX, PAN_INTEGRAL_MAX);
        tilt_integral = clampf(tilt_integral, -TILT_INTEGRAL_MAX, TILT_INTEGRAL_MAX);

        float pan_derivative = (pan_error - last_x_error) / dt;
        float tilt_derivative = (tilt_error - last_y_error) / dt;

        float pan_output_raw =
            KP_PAN * pan_error +
            KI_PAN * pan_integral +
            KD_PAN * pan_derivative;

        float tilt_output_raw =
            KP_TILT * tilt_error +
            KI_TILT * tilt_integral +
            KD_TILT * tilt_derivative;

        float pan_output = clampf(pan_output_raw, -PAN_STEP_MAX, PAN_STEP_MAX);
        float tilt_output = clampf(tilt_output_raw, -TILT_STEP_MAX, TILT_STEP_MAX);

        if (pan_output != pan_output_raw) {
            pan_integral -= pan_error * dt;
        }
        if (tilt_output != tilt_output_raw) {
            tilt_integral -= tilt_error * dt;
        }

        pan_angle += pan_output;
        tilt_angle -= tilt_output;

        last_x_error = pan_error;
        last_y_error = tilt_error;
    }

    pan_angle = clampf(pan_angle, PAN_MIN, PAN_MAX);
    tilt_angle = clampf(tilt_angle, TILT_MIN, TILT_MAX);

    send_servo_targets();
}

static void handle_anim(const char *animation, const char *text) {
    printf("[ANIM] animation=%-10s text=%s\n", animation, text ? text : "");

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

/* ==============================================================================
 * JSON Parsing Helpers
 * ============================================================================== */

static int json_str(const char *json, const char *key, char *out, int out_sz) {
    char search[64];
    snprintf(search, sizeof(search), "\"%s\"", key);

    const char *p = strstr(json, search);
    if (!p) return 0;

    p += strlen(search);
    while (*p == ' ' || *p == ':') p++;

    if (*p == '\"') {
        p++;
        int i = 0;
        while (*p && *p != '\"' && i < out_sz - 1) {
            out[i++] = *p++;
        }
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
    if (!json_str(line, "type", type, sizeof(type))) {
        return;
    }

    if (strcmp(type, "pos") == 0) {
        float x = 0.0f, y = 0.0f, conf = 0.0f;
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

/* ==============================================================================
 * IPC Server & Main Routine
 * ============================================================================== */

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
                if (line_len > 0) {
                    dispatch(line);
                }
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
    printf(" ║         Arduino Serial Interface Setup      ║\n");
    printf(" ╚══════════════════════════════════════════════╝\n\n");

    FILE *fp;
    char path[1035];
    char ports[20][256];
    int port_count = 0;

    fp = popen("ls -1 /dev/ttyACM* /dev/ttyUSB* 2>/dev/null", "r");
    if (fp == NULL) {
        printf("Failed to run ls command\n");
        return;
    }

    printf(" Available serial ports:\n");
    while (fgets(path, sizeof(path), fp) != NULL && port_count < 20) {
        path[strcspn(path, "\r\n")] = '\0';
        strncpy(ports[port_count], path, sizeof(ports[port_count]) - 1);
        ports[port_count][sizeof(ports[port_count]) - 1] = '\0';
        port_count++;
        printf(" [%d] %s\n", port_count, path);
    }

    pclose(fp);

    if (port_count == 0) {
        printf("\n [Serial] No serial ports found. Running without Arduino.\n");
        return;
    }

    printf("\n Enter device number (1-%d) or press Enter to skip:\n", port_count);
    printf(" > ");
    fflush(stdout);

    char input[256];
    if (!fgets(input, sizeof(input), stdin) || input[0] == '\n' || input[0] == '\0') {
        printf("[Serial] Skipped serial setup. Running without Arduino.\n");
        return;
    }

    int choice = atoi(input);
    if (choice > 0 && choice <= port_count) {
        strncpy(saved_port, ports[choice - 1], sizeof(saved_port) - 1);
        saved_port[sizeof(saved_port) - 1] = '\0';
    } else {
        printf("[Serial] Invalid selection. Running without Arduino.\n");
        return;
    }

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