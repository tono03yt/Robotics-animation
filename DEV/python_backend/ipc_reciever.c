/*
 * ipc_receiver.c — Unix Domain Socket receiver for the Python face-tracking backend
 *
 * Compile:  gcc -o ipc_receiver ipc_receiver.c
 * Usage:    ./ipc_receiver [socket_path]
 *           ./ipc_receiver                          → uses /tmp/robot_pipeline.sock
 *           ./ipc_receiver /tmp/my_custom.sock      → uses custom path
 *
 * Messages received (newline-delimited JSON):
 *   {"type":"pos",  "x":0.12, "y":-0.05, "conf":0.91}
 *   {"type":"anim", "animation":"speech", "text":"Hallo!"}
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>

/* ── configuration ─────────────────────────────────────────────────────────── */
#define DEFAULT_SOCK_PATH  "/tmp/robot_pipeline.sock"
#define BACKLOG            8
#define BUF_SIZE           4096

/* ── globals (for signal handler) ─────────────────────────────────────────── */
static int         g_server_fd  = -1;
static const char *g_sock_path  = NULL;

/* ── helpers ───────────────────────────────────────────────────────────────── */
static void cleanup(void)
{
    if (g_server_fd >= 0) {
        close(g_server_fd);
        g_server_fd = -1;
    }
    if (g_sock_path) {
        unlink(g_sock_path);
        printf("\n[IPC] Socket %s removed.\n", g_sock_path);
    }
}

static void on_signal(int sig)
{
    (void)sig;
    cleanup();
    exit(0);
}

/* ── message handlers — replace these with your robot logic ───────────────── */
static void handle_pos(float x, float y, float conf)
{
    /* TODO: drive servos / send to motion controller */
    printf("[POS]  x=%-8.4f  y=%-8.4f  conf=%.2f\n", x, y, conf);
}

static void handle_anim(const char *animation, const char *text)
{
    /* TODO: trigger facial animation on the robot */
    printf("[ANIM] animation=%-10s  text=%s\n", animation, text);
}

/* ── tiny JSON field extractor (no dependencies) ───────────────────────────── */
static int json_str(const char *json, const char *key, char *out, int out_sz)
{
    /* finds "key":"value" and copies value into out */
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

static int json_float(const char *json, const char *key, float *out)
{
    char search[64];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *p = strstr(json, search);
    if (!p) return 0;
    p += strlen(search);
    while (*p == ' ' || *p == ':') p++;
    *out = strtof(p, NULL);
    return 1;
}

/* ── process one JSON line ──────────────────────────────────────────────────── */
static void dispatch(const char *line)
{
    char type[32] = {0};
    if (!json_str(line, "type", type, sizeof(type))) return;

    if (strcmp(type, "pos") == 0) {
        float x = 0, y = 0, conf = 0;
        json_float(line, "x",    &x);
        json_float(line, "y",    &y);
        json_float(line, "conf", &conf);
        handle_pos(x, y, conf);

    } else if (strcmp(type, "anim") == 0) {
        char animation[64] = {0};
        char text[512]     = {0};
        json_str(line, "animation", animation, sizeof(animation));
        json_str(line, "text",      text,      sizeof(text));
        handle_anim(animation, text);

    } else {
        printf("[IPC] Unknown message type: %s\n", type);
    }
}

/* ── read lines from a connected client ────────────────────────────────────── */
static void serve_client(int client_fd)
{
    char  buf[BUF_SIZE];
    char  line[BUF_SIZE];
    int   line_len = 0;
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
    /* flush partial line if connection dropped mid-message */
    if (line_len > 0) {
        line[line_len] = '\0';
        dispatch(line);
    }
    close(client_fd);
}

/* ── interactive socket path selection ─────────────────────────────────────── */
static const char *select_socket_path(int argc, char *argv[])
{
    if (argc >= 2) {
        printf("[IPC] Using socket path from argument: %s\n", argv[1]);
        return argv[1];
    }

    printf("\n");
    printf("  ╔══════════════════════════════════════════════╗\n");
    printf("  ║     IPC Receiver — Socket Path Setup         ║\n");
    printf("  ╚══════════════════════════════════════════════╝\n\n");
    printf("  Default path : %s\n\n", DEFAULT_SOCK_PATH);
    printf("  Enter socket path (or press Enter for default):\n");
    printf("  > ");
    fflush(stdout);

    static char input[256];
    if (!fgets(input, sizeof(input), stdin) || input[0] == '\n' || input[0] == '\0') {
        printf("[IPC] Using default: %s\n", DEFAULT_SOCK_PATH);
        return DEFAULT_SOCK_PATH;
    }

    /* strip newline */
    input[strcspn(input, "\r\n")] = '\0';
    printf("[IPC] Using: %s\n", input);
    return input;
}

/* ── main ───────────────────────────────────────────────────────────────────── */
int main(int argc, char *argv[])
{
    /* select path */
    g_sock_path = select_socket_path(argc, argv);

    /* register cleanup */
    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);

    /* create socket */
    g_server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (g_server_fd < 0) {
        perror("[IPC] socket()");
        return 1;
    }

    /* remove stale socket file */
    unlink(g_sock_path);

    /* bind */
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
    printf("[IPC]   Start the Python backend now — it will auto-connect.\n");
    printf("[IPC]   Press Ctrl+C to quit.\n\n");

    /* accept loop */
    while (1) {
        int client_fd = accept(g_server_fd, NULL, NULL);
        if (client_fd < 0) {
            if (errno == EINTR) break;   /* signal received */
            perror("[IPC] accept()");
            continue;
        }
        printf("[IPC] Python connected.\n");
        serve_client(client_fd);
        printf("[IPC] Python disconnected — waiting for reconnect...\n");
    }

    cleanup();
    return 0;
}