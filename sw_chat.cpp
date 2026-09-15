#include <iostream>
#include <unordered_set>
#include <iomanip>
#include <sys/ioctl.h>
#include <unistd.h>
#include <fstream>
#include <cstdio>
#include <cctype>
#include <wchar.h>
#include <unordered_map>
#include <sys/stat.h>
#include <dirent.h>
#include <fcntl.h>
#include <poll.h>
#include <cerrno>
#include <sys/types.h>
#include <algorithm>
#include <string>
#include <vector>
#include <signal.h>
#include <sys/wait.h>
#include <sstream>
#include <atomic>
#include <thread>
#include <chrono>
#include <ctime>
#include <mutex>
#include <iterator>
#include <cstring>
#include <clocale>
#include <readline/readline.h>
#include <readline/history.h>
#include <curl/curl.h>
#include "nlohmann/json.hpp"

using json = nlohmann::json;
// ─────────────────────────── Версия ───────────────────────────
#define APP_VERSION "1.4.11"


// Emoji_Presentation: всегда отображается как emoji (ширина 2)
struct CpRange { uint32_t lo, hi; };
static bool cp_in_ranges(int cp, const CpRange* r, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (cp < static_cast<int>(r[i].lo)) return false;
        if (cp <= static_cast<int>(r[i].hi)) return true;
    }
    return false;
}
static const CpRange EMOJI_PRES[] = {
    {0x231A, 0x231B}, {0x23E9, 0x23EC}, {0x23F0, 0x23F0}, {0x23F3, 0x23F3},
    {0x25FD, 0x25FE}, {0x2614, 0x2615}, {0x2648, 0x2653}, {0x267F, 0x267F},
    {0x2693, 0x2693}, {0x26A1, 0x26A1}, {0x26AA, 0x26AB}, {0x26BD, 0x26BE},
    {0x26C4, 0x26C5}, {0x26CE, 0x26CE}, {0x26D4, 0x26D4}, {0x26EA, 0x26EA},
    {0x26F2, 0x26F3}, {0x26F5, 0x26F5}, {0x26FA, 0x26FA}, {0x26FD, 0x26FD},
    {0x2705, 0x2705}, {0x270A, 0x270B}, {0x2728, 0x2728}, {0x274C, 0x274C},
    {0x274E, 0x274E}, {0x2753, 0x2755}, {0x2757, 0x2757}, {0x2795, 0x2797},
    {0x27B0, 0x27B0}, {0x27BF, 0x27BF}, {0x2B1B, 0x2B1C}, {0x2B50, 0x2B50},
    {0x2B55, 0x2B55}, {0x1F004, 0x1F004}, {0x1F0CF, 0x1F0CF}, {0x1F18E, 0x1F18E},
    {0x1F191, 0x1F19A}, {0x1F1E6, 0x1F1FF}, {0x1F201, 0x1F201}, {0x1F21A, 0x1F21A},
    {0x1F22F, 0x1F22F}, {0x1F232, 0x1F236}, {0x1F238, 0x1F23A}, {0x1F250, 0x1F251},
    {0x1F300, 0x1F320}, {0x1F32D, 0x1F335}, {0x1F337, 0x1F37C}, {0x1F37E, 0x1F393},
    {0x1F3A0, 0x1F3CA}, {0x1F3CF, 0x1F3D3}, {0x1F3E0, 0x1F3F0}, {0x1F3F4, 0x1F3F4},
    {0x1F3F8, 0x1F43E}, {0x1F440, 0x1F440}, {0x1F442, 0x1F4FC}, {0x1F4FF, 0x1F53D},
    {0x1F54B, 0x1F54E}, {0x1F550, 0x1F567}, {0x1F57A, 0x1F57A}, {0x1F595, 0x1F596},
    {0x1F5A4, 0x1F5A4}, {0x1F5FB, 0x1F64F}, {0x1F680, 0x1F6C5}, {0x1F6CC, 0x1F6CC},
    {0x1F6D0, 0x1F6D2}, {0x1F6D5, 0x1F6D7}, {0x1F6DD, 0x1F6DF}, {0x1F6EB, 0x1F6EC},
    {0x1F6F4, 0x1F6FC}, {0x1F7E0, 0x1F7EB}, {0x1F7F0, 0x1F7F0}, {0x1F90C, 0x1F93A},
    {0x1F93C, 0x1F945}, {0x1F947, 0x1F9FF}, {0x1FA70, 0x1FA74}, {0x1FA78, 0x1FA7C},
    {0x1FA80, 0x1FA86}, {0x1FA90, 0x1FAAC}, {0x1FAB0, 0x1FABA}, {0x1FAC0, 0x1FAC5},
    {0x1FAD0, 0x1FAD9}, {0x1FAE0, 0x1FAE7}, {0x1FAF0, 0x1FAF6},
};
static const CpRange EMOJI_CODE[] = {
    {0x0023, 0x0023}, {0x002A, 0x002A}, {0x0030, 0x0039}, {0x00A9, 0x00A9},
    {0x00AE, 0x00AE}, {0x203C, 0x203C}, {0x2049, 0x2049}, {0x2122, 0x2122},
    {0x2139, 0x2139}, {0x2194, 0x2199}, {0x21A9, 0x21AA}, {0x231A, 0x231B},
    {0x2328, 0x2328}, {0x23CF, 0x23CF}, {0x23E9, 0x23F3}, {0x23F8, 0x23FA},
    {0x24C2, 0x24C2}, {0x25AA, 0x25AB}, {0x25B6, 0x25B6}, {0x25C0, 0x25C0},
    {0x25FB, 0x25FE}, {0x2600, 0x2604}, {0x260E, 0x260E}, {0x2611, 0x2611},
    {0x2614, 0x2615}, {0x2618, 0x2618}, {0x261D, 0x261D}, {0x2620, 0x2620},
    {0x2622, 0x2623}, {0x2626, 0x2626}, {0x262A, 0x262A}, {0x262E, 0x262F},
    {0x2638, 0x263A}, {0x2640, 0x2640}, {0x2642, 0x2642}, {0x2648, 0x2653},
    {0x265F, 0x2660}, {0x2663, 0x2663}, {0x2665, 0x2666}, {0x2668, 0x2668},
    {0x267B, 0x267B}, {0x267E, 0x267F}, {0x2692, 0x2697}, {0x2699, 0x2699},
    {0x269B, 0x269C}, {0x26A0, 0x26A1}, {0x26A7, 0x26A7}, {0x26AA, 0x26AB},
    {0x26B0, 0x26B1}, {0x26BD, 0x26BE}, {0x26C4, 0x26C5}, {0x26C8, 0x26C8},
    {0x26CE, 0x26CF}, {0x26D1, 0x26D1}, {0x26D3, 0x26D4}, {0x26E9, 0x26EA},
    {0x26F0, 0x26F5}, {0x26F7, 0x26FA}, {0x26FD, 0x26FD}, {0x2702, 0x2702},
    {0x2705, 0x2705}, {0x2708, 0x270D}, {0x270F, 0x270F}, {0x2712, 0x2712},
    {0x2714, 0x2714}, {0x2716, 0x2716}, {0x271D, 0x271D}, {0x2721, 0x2721},
    {0x2728, 0x2728}, {0x2733, 0x2734}, {0x2744, 0x2744}, {0x2747, 0x2747},
    {0x274C, 0x274C}, {0x274E, 0x274E}, {0x2753, 0x2755}, {0x2757, 0x2757},
    {0x2763, 0x2764}, {0x2795, 0x2797}, {0x27A1, 0x27A1}, {0x27B0, 0x27B0},
    {0x27BF, 0x27BF}, {0x2934, 0x2935}, {0x2B05, 0x2B07}, {0x2B1B, 0x2B1C},
    {0x2B50, 0x2B50}, {0x2B55, 0x2B55}, {0x3030, 0x3030}, {0x303D, 0x303D},
    {0x3297, 0x3297}, {0x3299, 0x3299}, {0x1F004, 0x1F004}, {0x1F0CF, 0x1F0CF},
    {0x1F170, 0x1F171}, {0x1F17E, 0x1F17F}, {0x1F18E, 0x1F18E}, {0x1F191, 0x1F19A},
    {0x1F1E6, 0x1F1FF}, {0x1F201, 0x1F202}, {0x1F21A, 0x1F21A}, {0x1F22F, 0x1F22F},
    {0x1F232, 0x1F23A}, {0x1F250, 0x1F251}, {0x1F300, 0x1F321}, {0x1F324, 0x1F393},
    {0x1F396, 0x1F397}, {0x1F399, 0x1F39B}, {0x1F39E, 0x1F3F0}, {0x1F3F3, 0x1F3F5},
    {0x1F3F7, 0x1F4FD}, {0x1F4FF, 0x1F53D}, {0x1F549, 0x1F54E}, {0x1F550, 0x1F567},
    {0x1F56F, 0x1F570}, {0x1F573, 0x1F57A}, {0x1F587, 0x1F587}, {0x1F58A, 0x1F58D},
    {0x1F590, 0x1F590}, {0x1F595, 0x1F596}, {0x1F5A4, 0x1F5A5}, {0x1F5A8, 0x1F5A8},
    {0x1F5B1, 0x1F5B2}, {0x1F5BC, 0x1F5BC}, {0x1F5C2, 0x1F5C4}, {0x1F5D1, 0x1F5D3},
    {0x1F5DC, 0x1F5DE}, {0x1F5E1, 0x1F5E1}, {0x1F5E3, 0x1F5E3}, {0x1F5E8, 0x1F5E8},
    {0x1F5EF, 0x1F5EF}, {0x1F5F3, 0x1F5F3}, {0x1F5FA, 0x1F64F}, {0x1F680, 0x1F6C5},
    {0x1F6CB, 0x1F6D2}, {0x1F6D5, 0x1F6D7}, {0x1F6DD, 0x1F6E5}, {0x1F6E9, 0x1F6E9},
    {0x1F6EB, 0x1F6EC}, {0x1F6F0, 0x1F6F0}, {0x1F6F3, 0x1F6FC}, {0x1F7E0, 0x1F7EB},
    {0x1F7F0, 0x1F7F0}, {0x1F90C, 0x1F93A}, {0x1F93C, 0x1F945}, {0x1F947, 0x1F9FF},
    {0x1FA70, 0x1FA74}, {0x1FA78, 0x1FA7C}, {0x1FA80, 0x1FA86}, {0x1FA90, 0x1FAAC},
    {0x1FAB0, 0x1FABA}, {0x1FAC0, 0x1FAC5}, {0x1FAD0, 0x1FAD9}, {0x1FAE0, 0x1FAE7},
    {0x1FAF0, 0x1FAF6},
};
static bool is_emoji_presentation(int cp) { return cp_in_ranges(cp, EMOJI_PRES, sizeof(EMOJI_PRES)/sizeof(EMOJI_PRES[0])); }
static bool is_emoji_codepoint(int cp) { return cp_in_ranges(cp, EMOJI_CODE, sizeof(EMOJI_CODE)/sizeof(EMOJI_CODE[0])); }

[[maybe_unused]] static int get_char_width(wchar_t wc) {
    int cp = static_cast<int>(wc);
    if (is_emoji_presentation(cp)) return 2;
    if (is_emoji_codepoint(cp)) return 1; // без VS16 — текстовый
    int w = wcwidth(wc);
    return (w > 0) ? w : 0;
}

// ─────────────────────────── Цвета ───────────────────────────
#define C_RESET   "\033[0m"
#define C_GREEN   "\033[32m"
#define C_INPUT   "\033[96m"   // цвет текста, вводимого пользователем
#define C_CYAN    "\033[36m"
#define C_YELLOW  "\033[33m"
#define C_RED     "\033[31m"
#define C_BLUE    "\033[34m"
#define C_BOLD    "\033[1m"
#define C_GRAY    "\033[90m"
#define C_MAGENTA "\033[35m"
#define C_ITALIC  "\033[3m"
#define C_BG_GRAY "\033[48;5;236m"
#define C_WHITE   "\033[97m"
#define C_CODE_FG "\033[93m"
#define C_QUOTE   "\033[36;3m"
#define C_BULLET  "\033[33m"
#define C_H1      "\033[1;35m"
#define C_H2      "\033[1;36m"
#define C_H3      "\033[1;33m"

// ─────────────────────────── Вспомогательная функция ─────────
// Заменяет emoji-флаги (пары Regional Indicator) на текстовый код [XX]
static std::string replace_flags(const std::string &s) {
    mbtowc(nullptr, nullptr, 0); // Сброс сдвига UTF-8
    std::string result;
    result.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        wchar_t wc = 0;
        int clen = mbtowc(&wc, s.c_str() + i, MB_CUR_MAX);
        if (clen <= 0) { result += s[i++]; continue; }
        int cp = static_cast<int>(wc);
        if (cp >= 0x1F1E0 && cp <= 0x1F1FF) {
            // Первая буква флага
            char letter1 = static_cast<char>('A' + (cp - 0x1F1E6));
            i += clen;
            if (i < s.size()) {
                wchar_t next_cp = 0;
                int next_clen = mbtowc(&next_cp, s.c_str() + i, MB_CUR_MAX);
                if (next_clen > 0 && next_cp >= 0x1F1E0 && next_cp <= 0x1F1FF) {
                    char letter2 = static_cast<char>('A' + (next_cp - 0x1F1E6));
                    result += '[';
                    result += letter1;
                    result += letter2;
                    result += ']';
                    i += next_clen;
                    continue;
                }
            }
            result += '[';
            result += letter1;
            result += ']';
            continue;
        }
        result.append(s, i, clen);
        i += clen;
    }
    return result;
}

static std::string get_home_dir() {
    const char* h = getenv("HOME");
    return h ? std::string(h) : "/tmp";
}

// ─────────────────────────── Константы ───────────────────────
#define CMD_TIMEOUT         250
#define MAX_CMD_OUTPUT      50000
#define MAX_FILE_BYTES      200000
#define MAX_MSG_CHARS       120000
#define HISTORY_SAVE_EVERY  8
#define MODELS_PAGE_SIZE    25
#define MODELS_MAX_CACHE    2000
#define RL_HIST_MAX_CHARS   2000
#define MAX_BASH_CHAIN      7

#define MAX_MESSAGES        500
#define DEFAULT_TEMPERATURE 0.7
#define DEFAULT_MAX_TOKENS  4096
#define DEFAULT_API_BASE    "https://api.302.ai"

static std::string HISTORY_FILE;
static std::string SYSTEM_PROMPT_FILE;
static std::string READLINE_HIST_FILE;
static std::string CONFIG_DIR;
static std::string CONFIG_FILE;
static std::string SESSIONS_DIR;

// ─────────────────────────── Глобальное состояние ────────────
struct ChatSession {
    std::vector<json> messages;
    std::string       history_file;
    // ****************   модель по умолчанию ******************
    //
    //std::string       model          = "nvidia/nemotron-3-super-120b-a12b:free";
    //std::string       model          = "minimax/minimax-m2.7";
    //std::string       model          = "anthropic/claude-sonnet-4";
    //std::string       model          = "openai/gpt-5.4";
    //std::string       model          = "google/gemini-3.1-pro-preview";
    //std::string       model          = "x-ai/grok-4.20-beta";
    //std::string       model          = "qwen/qwen3.5-397b-a17b";
    //std::string       model          = "qwen/qwen3.6-plus:free";
    //std::string       model          = "xiaomi/mimo-v2-flash";
    //std::string       model          = "xiaomi/mimo-v2-pro"
    std::string       model          = "deepseek-chat";
    //std::string       model          = "anthropic/claude-opus-4.8";
    //std::string       model          = "~google/gemini-pro-latest";
    //std::string       model          = "~anthropic/claude-sonnet-latest";
    //std::string       model          = "qwen-max";
    //std::string       model          = "anthropic/claude-sonnet-4.6";

    std::string       sys_prompt;
    double            temperature    = DEFAULT_TEMPERATURE;
    int               max_tokens     = DEFAULT_MAX_TOKENS;
    int               total_prompt_tokens     = 0;
    int               total_completion_tokens = 0;
    bool              autorun                 = false;
    bool              history_enabled          = false;
    bool              nores                    = false; // выкл по умолчанию
    bool              compact_mode             = false;
    bool              fire                     = false; // FIRE = silent_bash + autorun
    bool              time_prefix              = false; // префикс даты-времени МСК в начале запроса
    bool              voice_in                 = false; // STT: termux-speech-to-text
    bool              voice_out                = false; // TTS: termux-tts-speak
    std::string       voice_lang;                       // -l (пусто = системный)
    std::string       voice_region;                     // -n
    std::string       voice_variant;                    // -v
    std::string       voice_engine;                     // -e (пусто = default)
    std::string       voice_stream            = "NOTIFICATION"; // -s
    double            voice_pitch             = 1.0;    // -p
    double            voice_rate              = 1.0;    // -r
    bool              voice_short             = false;  // озвучивать только первый абзац
    std::string       session_name             = "default";
    std::string       api_base                 = DEFAULT_API_BASE;
    std::unordered_map<std::string, std::string> aliases;
};

static ChatSession G;

// ───────── /undo снапшоты ─────────
static std::vector<std::vector<json>> g_undo_stack;
static const size_t UNDO_MAX = 20;
static void push_undo_snapshot() {
    g_undo_stack.push_back(G.messages);
    if (g_undo_stack.size() > UNDO_MAX)
        g_undo_stack.erase(g_undo_stack.begin());
}

// Fallback-список, если API/кэш недоступны
static const std::vector<std::string> DEFAULT_MODELS = {
    "claude-sonnet-5",
    "claude-fable-5",
    "gpt-5.6-terra",
    "gpt-5.6-sol",
    "gemini-3.1-pro-preview",
    "grok-4.5",
    "deepseek-reasoner",
    "deepseek-chat",
    "qwen3.7-max",
    "qwen3.7-plus",
    "MiniMax-M3",
    "glm-5.2",
    "kimi-k2.7-code",
    "doubao-seed-2-1-turbo-260628",
    "doubao-seed-2-1-pro-260628"
};
// Живой список (кэш/API); при старте = DEFAULT_MODELS
static std::vector<std::string> AVAILABLE_MODELS = DEFAULT_MODELS;
// Последний показанный (отфильтрованный) список — /model N берёт номера отсюда
static std::vector<std::string> LAST_MODEL_VIEW;
static std::string MODELS_CACHE_FILE;
// live pricing from /v1/models when API provides it: model -> (prompt$/MTok, completion$/MTok)
static std::unordered_map<std::string, std::pair<double,double>> MODEL_PRICING_LIVE;

// ─────────────────────────── Сигналы ─────────────────────────
// g_exit_requested: 1 = выход из программы
// g_stream_abort:   1 = прервать текущий стриминг (Ctrl+C во время ответа)
static volatile sig_atomic_t g_exit_requested = 0;
static volatile sig_atomic_t g_stream_abort   = 0;
static volatile sig_atomic_t g_in_streaming   = 0; // 1 пока идёт стриминг API
static volatile sig_atomic_t g_in_bash        = 0; // 1 пока выполняется bash-блок
static volatile sig_atomic_t g_last_sig_time  = 0; // время последнего Ctrl+C
static bool g_dry_run = false;                     // --dry-run: не выполнять bash-блоки
static std::mutex g_stream_mutex;

// Логика Ctrl+C:
//  * во время операции (curl/bash/пауза) первый Ctrl+C прерывает её;
//  * повторный Ctrl+C в течение 2 сек — выход из программы;
//  * вне операции — выход из программы.
// Только async-signal-safe действия (time(), запись sig_atomic_t).
static void signal_handler(int /*sig*/) {
    time_t now = time(nullptr);
    bool double_press = (g_last_sig_time != 0 &&
                         static_cast<long>((now - static_cast<time_t>(g_last_sig_time))) <= 2);
    g_last_sig_time = static_cast<sig_atomic_t>(now);
    if (g_in_streaming || g_in_bash) {
        g_stream_abort = 1;                      // прервать текущую операцию
        if (double_press) g_exit_requested = 1;  // ...и выйти из программы
    } else {
        g_exit_requested = 1;
    }
}

// ─────────────────────────── API ключ ────────────────────────
static std::string get_api_key() {
    const char* env = getenv("302_API_KEY");
    if (env && std::string(env).size() > 10) return std::string(env);
    std::string home = get_home_dir();
    std::string keyfile = home + "/.config/302_key";

    // P0.3: предупреждаем, если файл ключа доступен группе/остальным, и пробуем
    // исправить права автоматически (chmod 600).
    {
        struct stat st;
        if (stat(keyfile.c_str(), &st) == 0) {
            if ((st.st_mode & (S_IRWXG | S_IRWXO)) != 0) {
                std::cerr << C_YELLOW
                          << "[WARN: " << keyfile << " доступен группе/остальным (mode "
                          << std::oct << (st.st_mode & 0777) << std::dec
                          << "). Исправляю на 600...]"
                          << C_RESET << std::endl;
                chmod(keyfile.c_str(), 0600);
            }
        }
    }

    std::ifstream f(keyfile);
    if (f.is_open()) {
        std::string key;
        std::getline(f, key);
        while (!key.empty() && (key.back() == '\n' || key.back() == '\r' || key.back() == ' '))
            key.pop_back();
        if (key.size() > 10) return key;
    }
    std::cerr << C_RED << "[ОШИБКА: API ключ не найден!]" << C_RESET << std::endl;
    return "";
}

// ─────────────────────────── UTF-8 ───────────────────────────
std::string sanitize_utf8(const std::string &input) {
    std::string result;
    result.reserve(input.size());
    size_t i = 0;
    while (i < input.size()) {
        unsigned char c = input[i];
        int len = 0;
        if      (c <= 0x7F)               len = 1;
        else if ((c & 0xE0) == 0xC0)      len = 2;
        else if ((c & 0xF0) == 0xE0)      len = 3;
        else if ((c & 0xF8) == 0xF0)      len = 4;
        else { ++i; continue; }
        if (i + static_cast<size_t>(len) > input.size()) break;
        bool valid = true;
        for (int j = 1; j < len; ++j)
            if ((input[i+j] & 0xC0) != 0x80) { valid = false; break; }
        if (valid) { result.append(input, i, len); i += len; }
        else ++i;
    }
    return result;
}

// ─────────────────────────── Markdown рендер ─────────────────
static std::string render_inline_md(const std::string &line) {
    std::string out;
    out.reserve(line.size() * 2);
    size_t i = 0, len = line.size();
    while (i < len) {
        // Инлайн-код
        if (line[i] == '`' && (i+1 < len) && line[i+1] != '`') {
            size_t end = line.find('`', i+1);
            if (end != std::string::npos) {
                out += C_BG_GRAY; out += C_CODE_FG;
                out += line.substr(i+1, end-i-1);
                out += C_RESET;
                i = end + 1; continue;
            }
        }
        // **bold**
        if (i+1 < len && line[i] == '*' && line[i+1] == '*') {
            size_t end = line.find("**", i+2);
            if (end != std::string::npos) {
                out += C_BOLD;
                out += line.substr(i+2, end-i-2);
                out += C_RESET;
                i = end + 2; continue;
            }
        }
        // __bold__
        if (i+1 < len && line[i] == '_' && line[i+1] == '_') {
            size_t end = line.find("__", i+2);
            if (end != std::string::npos) {
                out += C_BOLD;
                out += line.substr(i+2, end-i-2);
                out += C_RESET;
                i = end + 2; continue;
            }
        }
        // *italic* — skip if space after opening or before closing *
        if (line[i] == '*' && i+1 < len && line[i+1] != '*' && line[i+1] != ' ') {
            size_t end = line.find('*', i+1);
            if (end != std::string::npos && end > i+1 && line[end-1] != ' '
                && (end+1 >= len || line[end+1] != '*')) {
                out += C_ITALIC;
                out += line.substr(i+1, end-i-1);
                out += C_RESET;
                i = end + 1; continue;
            }
        }
        out += line[i]; ++i;
    }
    return out;
}

// ─────────────────────────── Markdown таблицы ─────────────────
static std::vector<std::string> split_table_cells(const std::string &line) {
    std::vector<std::string> cells;
    std::string trimmed = line;
    while (!trimmed.empty() && trimmed.front() == '|') trimmed.erase(0, 1);
    while (!trimmed.empty() && trimmed.back() == '|') trimmed.pop_back();
    std::istringstream ss(trimmed);
    std::string cell;
    while (std::getline(ss, cell, '|')) {
        size_t start = cell.find_first_not_of(' ');
        size_t end = cell.find_last_not_of(' ');
        if (start != std::string::npos)
            cells.push_back(cell.substr(start, end - start + 1));
        else
            cells.push_back("");
    }
    return cells;
}

// Визуальная ширина UTF-8 строки (без ANSI escape)
// Визуальная ширина UTF-8 строки (без ANSI escape)
// Фильтрует zero-width и combining символы (U+FE0F, U+200D и т.д.)
// Визуальная ширина UTF-8 строки (без ANSI escape)
// Проверка: является ли codepoint emoji (требует VS для отображения как emoji)
static size_t visible_width(const std::string &s) {
    mbtowc(nullptr, nullptr, 0); // Сброс сдвига UTF-8
    // 1. Убираем ANSI escape-последовательности (CSI и простые)
    std::string stripped;
    stripped.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        if (s[i] == '\033') {
            ++i;
            if (i < s.size() && s[i] == '[') {
                ++i;
                while (i < s.size() && !((s[i] >= '@' && s[i] <= '~'))) ++i;
                if (i < s.size()) ++i;
            } else if (i < s.size()) {
                ++i;
            }
            continue;
        }
        stripped += s[i++];
    }
    
    // 2. Декодируем UTF-8, считаем ширину с учётом emoji и variation selectors
    size_t w = 0;
    i = 0;
    while (i < stripped.size()) {
        wchar_t wc = 0;
        int clen = mbtowc(&wc, stripped.c_str() + i, MB_CUR_MAX);
        if (clen <= 0) { i++; continue; }

        // Ширина символа = wcwidth(codepoint).
        // Замерено позицией курсора (\e[6n) прямо в Termux — совпадает для ВСЕХ
        // эмодзи и комбинированных последовательностей:
        //   VS16:   ☁️  = wcwidth(U+2601)+wcwidth(U+FE0F) = 1+0 = 1
        //   VS16:   ⛅️  = wcwidth(U+26C5)+wcwidth(U+FE0F) = 2+0 = 2
        //   ZWJ:    👨🇳? = 👨+ZWJ+👩+ZWJ+👧 = 2+0+2+0+2 = 6 (Termux НЕ склеивает)
        //   тон:    👍🏽 = wcwidth(U+1F44D)+wcwidth(U+1F3FD) = 2+2 = 4 (skin tone НЕ zero)
        //   флаг:   🇷🇺 = 1+1 = 2
        //   keycap: 1️⃣ = 1+0+0 = 1
        // Вариационные селекторы/ZWJ/combining имеют wcwidth=0 и сами дают 0.
        // Спец-таблицы EMOJI_PRES/EMOJI_CODE больше НЕ используются.
        int char_w = wcwidth(wc);
        if (char_w > 0) w += static_cast<size_t>(char_w);
        i += clen;
    }
    return w;
}

// ── Терминал / word-wrap ─────────────────────────────────────
static bool is_compact();  // определена ниже
static int get_terminal_width() {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0)
        return static_cast<int>(ws.ws_col);
    return 80;
}

// Жадный word-wrap plain-текста по визуальной ширине (UTF-8 aware).
static std::vector<std::string> wrap_plain(const std::string& s, size_t max_w) {
    std::vector<std::string> out;
    if (max_w < 8) max_w = 8;
    std::istringstream ss(s);
    std::string word, cur;
    size_t cur_w = 0;
    while (ss >> word) {
        size_t ww = visible_width(word);
        if (ww > max_w) {
            if (!cur.empty()) { out.push_back(cur); cur.clear(); cur_w = 0; }
            size_t i = 0, cw = 0;
            std::string chunk;
            while (i < word.size()) {
                unsigned char c = static_cast<unsigned char>(word[i]);
                int len = 1;
                if (c >= 0xF0) len = 4;
                else if (c >= 0xE0) len = 3;
                else if (c >= 0xC0) len = 2;
                if (i + static_cast<size_t>(len) > word.size()) len = 1;
                std::string tok = word.substr(i, len);
                int tw = static_cast<int>(visible_width(tok));
                if (cw + static_cast<size_t>(tw) > max_w && !chunk.empty()) {
                    out.push_back(chunk); chunk.clear(); cw = 0;
                }
                chunk += tok; cw += static_cast<size_t>(tw);
                i += static_cast<size_t>(len);
            }
            if (!chunk.empty()) out.push_back(chunk);
            continue;
        }
        size_t sep = cur.empty() ? 0 : 1;
        if (cur_w + sep + ww > max_w && !cur.empty()) {
            out.push_back(cur);
            cur = word; cur_w = ww;
        } else {
            if (!cur.empty()) { cur += ' '; cur_w += 1; }
            cur += word; cur_w += ww;
        }
    }
    if (!cur.empty()) out.push_back(cur);
    if (out.empty()) out.push_back("");
    return out;
}

// Обрезка строки, содержащей ANSI-эскейпы, до видимой ширины.
// ANSI-коды копируются без учёта ширины; обрезка идёт по видимым символам.
// truncated=true, если что-то отрезано (в конце добавлен C_RESET).
static std::string truncate_to_visible_ansi(const std::string& s, size_t max_w, bool& truncated) {
    std::string out;
    out.reserve(s.size());
    size_t w = 0, i = 0;
    bool cut = false;
    while (i < s.size()) {
        if (s[i] == '\033') {
            size_t start = i++;
            if (i < s.size() && s[i] == '[') {
                ++i;
                while (i < s.size() && !(s[i] >= '@' && s[i] <= '~')) ++i;
                if (i < s.size()) ++i;
            } else if (i < s.size()) {
                ++i;
            }
            out.append(s, start, i - start);
            continue;
        }
        unsigned char c = static_cast<unsigned char>(s[i]);
        int len = 1;
        if (c >= 0xF0) len = 4;
        else if (c >= 0xE0) len = 3;
        else if (c >= 0xC0) len = 2;
        if (i + static_cast<size_t>(len) > s.size()) len = 1;
        std::string tok = s.substr(i, static_cast<size_t>(len));
        size_t tw = visible_width(tok);
        if (w + tw > max_w) { cut = true; break; }
        out += tok; w += tw; i += static_cast<size_t>(len);
    }
    if (cut) out += C_RESET;
    truncated = cut;
    return out;
}

static void render_table_row(const std::vector<std::string> &cells, const std::vector<size_t> &col_widths, bool is_header = false) {
    std::cout << C_GRAY << "\xe2\x94\x82" << C_RESET;
    for (size_t i = 0; i < cells.size(); ++i) {
        size_t col_w = (i < col_widths.size()) ? col_widths[i] : 12;
        // Сначала накладываем ANSI-разметку (bold / inline-code), затем обрезаем
        // по видимой ширине — иначе markdown-теги ** и ` искажают расчёт паддинга.
        std::string rendered = render_inline_md(cells[i]);
        std::string shown;
        bool truncated = false;
        if (visible_width(rendered) > col_w) {
            size_t keep = (col_w > 1) ? (col_w - 1) : col_w;
            shown = truncate_to_visible_ansi(rendered, keep, truncated);
            shown += "\xe2\x80\xa6"; // …
            truncated = true;
        } else {
            shown = rendered;
        }
        size_t vis_w = visible_width(shown);
        size_t pad = (vis_w < col_w) ? (col_w - vis_w) : 0;
        std::cout << " ";
        if (is_header) std::cout << C_BOLD;
        std::cout << shown;
        if (is_header || truncated) std::cout << C_RESET;
        std::cout << std::string(pad, ' ') << " ";
        std::cout << C_GRAY << "\xe2\x94\x82" << C_RESET;
    }
    std::cout << "\n";
}

// ── Выравнивание ASCII/box-art внутри code-блоков ─────────────
// Модель рисует рамки, считая эмодзи шириной 2, но Termux рисует
// ☁️/🌡️/🌤️ (codepoint+VS16) шириной 1. Плюс модель сама косячит с
// паддингом. Границы рамки состоят только из box-drawing (ширина 1) —
// их длина = задумка модели. Подгоняем каждую строку рамки под общую
// ширину, добавляя/убирая хвостовые пробелы. Не-рамки не трогаем.
static size_t bxa_seq_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}
static int bxa_cp_at(const std::string &s, size_t i, size_t &len) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    len = bxa_seq_len(c);
    if (len == 1 || i + len > s.size()) { len = 1; return c; }
    int cp = 0;
    if (len == 2) cp = ((c & 0x1F) << 6) | (s[i+1] & 0x3F);
    else if (len == 3) cp = ((c & 0x0F) << 12) | ((s[i+1] & 0x3F) << 6) | (s[i+2] & 0x3F);
    else cp = ((c & 0x07) << 18) | ((s[i+1] & 0x3F) << 12) | ((s[i+2] & 0x3F) << 6) | (s[i+3] & 0x3F);
    return cp;
}
static bool bxa_is_box(int cp) { return cp >= 0x2500 && cp <= 0x257F; }

static std::vector<std::string> realign_box_art(const std::vector<std::string>& in) {
    // 1. Целевая ширина = максимальная ширина строки из чистых box-drawing.
    size_t target = 0;
    for (auto &l : in) {
        if (l.empty()) continue;
        bool all = true; size_t i = 0;
        while (i < l.size()) { size_t n; int cp = bxa_cp_at(l, i, n);
            if (!bxa_is_box(cp)) { all = false; break; } i += n; }
        if (all) { size_t w = visible_width(l); if (w > target) target = w; }
    }
    // 2. Считаем, сколько строк "в рамке" (начинаются и кончаются box-drawing).
    size_t framed = 0;
    for (auto &l : in) {
        if (l.size() < 2) continue;
        size_t n1; int a = bxa_cp_at(l, 0, n1);
        size_t last = 0, i = 0; while (i < l.size()) { size_t n; bxa_cp_at(l, i, n); last = i; i += n; }
        size_t n2; int b = bxa_cp_at(l, last, n2);
        if (bxa_is_box(a) && bxa_is_box(b)) ++framed;
    }
    bool art = (framed >= 2 && target > 0);
    std::vector<std::string> out;
    for (auto &l : in) {
        if (!art || l.size() < 2) { out.push_back(l); continue; }
        size_t n1; int a = bxa_cp_at(l, 0, n1);
        size_t last = 0, i = 0; while (i < l.size()) { size_t n; bxa_cp_at(l, i, n); last = i; i += n; }
        size_t n2; int b = bxa_cp_at(l, last, n2);
        if (!bxa_is_box(a) || !bxa_is_box(b)) { out.push_back(l); continue; }
        std::string first = l.substr(0, n1), lastc = l.substr(last, n2);
        std::string inner = l.substr(n1, last - n1);
        size_t e = inner.size(); while (e > 0 && inner[e-1] == ' ') --e;
        inner = inner.substr(0, e);
        size_t wf = visible_width(first), wl = visible_width(lastc);
        size_t tw = (target > wf + wl) ? (target - wf - wl) : 0;
        size_t iw = visible_width(inner);
        std::string res = first + inner;
        if (iw < tw) res += std::string(tw - iw, ' ');
        res += lastc;
        out.push_back(res);
    }
    return out;
}

// ── HTML-сущности ─────────────────────────────────────────────
// Модели любят вставлять &nbsp; &amp; &quot; &#8212; и т.п. «для красоты».
// В терминале они печатаются буквально и ломают вид. Раскрываем основные
// сущности в обычном тексте (в code-блоках НЕ трогаем — там это может
// быть настоящий пример HTML-кода).
static std::string unescape_html(const std::string &s) {
    if (s.find('&') == std::string::npos) return s;
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        if (s[i] != '&') { out += s[i++]; continue; }
        size_t semi = s.find(';', i + 1);
        if (semi == std::string::npos || semi - i > 12) { out += s[i++]; continue; }
        std::string ent = s.substr(i + 1, semi - i - 1);
        std::string rep;
        bool matched = true;
        if      (ent == "nbsp")   rep = " ";
        else if (ent == "amp")    rep = "&";
        else if (ent == "lt")     rep = "<";
        else if (ent == "gt")     rep = ">";
        else if (ent == "quot")   rep = "\"";
        else if (ent == "apos")   rep = "'";
        else if (ent == "copy")   rep = "\xc2\xa9";
        else if (ent == "reg")    rep = "\xc2\xae";
        else if (ent == "trade")  rep = "\xe2\x84\xa2";
        else if (ent == "hellip") rep = "\xe2\x80\xa6";
        else if (ent == "mdash")  rep = "\xe2\x80\x94";
        else if (ent == "ndash")  rep = "\xe2\x80\x93";
        else if (ent == "deg")    rep = "\xc2\xb0";
        else if (ent == "times")  rep = "\xc3\x97";
        else if (!ent.empty() && ent[0] == '#') {
            unsigned long cp = 0;
            bool ok = true;
            if (ent.size() > 1 && (ent[1] == 'x' || ent[1] == 'X')) {
                for (size_t k = 2; k < ent.size(); ++k) {
                    char c = ent[k]; int d;
                    if      (c >= '0' && c <= '9') d = c - '0';
                    else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
                    else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
                    else { ok = false; break; }
                    cp = cp * 16 + static_cast<unsigned long>(d);
                    if (cp > 0x10FFFF) { ok = false; break; }
                }
            } else {
                for (size_t k = 1; k < ent.size(); ++k) {
                    char c = ent[k];
                    if (c < '0' || c > '9') { ok = false; break; }
                    cp = cp * 10 + static_cast<unsigned long>(c - '0');
                    if (cp > 0x10FFFF) { ok = false; break; }
                }
            }
            if (ok && cp > 0 && cp <= 0x10FFFF &&
                !(cp >= 0xD800 && cp <= 0xDFFF) && cp != 0) {
                if (cp < 0x80) {
                    rep += static_cast<char>(cp);
                } else if (cp < 0x800) {
                    rep += static_cast<char>(0xC0 | (cp >> 6));
                    rep += static_cast<char>(0x80 | (cp & 0x3F));
                } else if (cp < 0x10000) {
                    rep += static_cast<char>(0xE0 | (cp >> 12));
                    rep += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                    rep += static_cast<char>(0x80 | (cp & 0x3F));
                } else {
                    rep += static_cast<char>(0xF0 | (cp >> 18));
                    rep += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
                    rep += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                    rep += static_cast<char>(0x80 | (cp & 0x3F));
                }
            } else {
                matched = false;
            }
        }
        else matched = false;
        if (matched) { out += rep; i = semi + 1; }
        else         { out += s[i++]; }
    }
    return out;
}

static void render_markdown(const std::string &text) {
    std::istringstream ss(text);
    std::string line;
    bool in_code = false;
    std::vector<std::string> code_lines;
    while (std::getline(ss, line)) {
        // Прерывание длинного вывода по Ctrl+C.
        if (g_stream_abort) {
            std::cout << "\n" << C_YELLOW << "[вывод прерван]" << C_RESET << std::endl;
            break;
        }
        if (!line.empty() && line.back() == '\r') line.pop_back();

        // Блок кода
        if (line.size() >= 3 && line.substr(0,3) == "```") {
            if (!in_code) {
                in_code = true;
                std::string lang = line.size() > 3 ? line.substr(3) : "code";
                while (!lang.empty() && lang[0]==' ') lang.erase(0,1);
                if (lang.empty()) lang = "code";
                std::cout << C_GRAY << "\xe2\x94\x8c\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80 "
                          << lang << " \xe2\x94\x80\xe2\x94\x80\xe2\x94\x80" << C_RESET << "\n";
            } else {
                in_code = false;
                for (auto &cl : realign_box_art(code_lines))
                    std::cout << C_GRAY << "\xe2\x94\x82 " << C_WHITE << cl << C_RESET << "\n";
                code_lines.clear();
                std::cout << C_GRAY
                          << "\xe2\x94\x94\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
                             "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
                             "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
                          << C_RESET << "\n";
            }
            continue;
        }
        if (in_code) {
            code_lines.push_back(line);
            continue;
        }
        // Раскрываем HTML-сущности (&nbsp; &amp; &#8212; и т.п.) в обычном тексте
        line = unescape_html(line);

        // Заголовки
        if (line.size()>=4 && line.substr(0,4)=="### ") {
            std::cout << C_H3 << "  \xe2\x96\xb8 " << line.substr(4) << C_RESET << "\n"; continue;
        }
        if (line.size()>=3 && line.substr(0,3)=="## ") {
            std::cout << C_H2 << " \xe2\x96\xb8 " << line.substr(3) << C_RESET << "\n"; continue;
        }
        if (line.size()>=2 && line.substr(0,2)=="# ") {
            std::cout << C_H1 << "\xe2\x96\xb8 " << line.substr(2) << C_RESET << "\n"; continue;
        }

        // Цитата
        if (!line.empty() && line[0]=='>') {
            std::string c = line.size()>1 ? line.substr(1) : "";
            if (!c.empty() && c[0]==' ') c.erase(0,1);
            if (is_compact()) {
                std::cout << "  \xe2\x94\x83 " << c << "\n";
            } else {
                int tw = get_terminal_width();
                size_t wrap_w = (tw > 12) ? static_cast<size_t>((tw - 4)) : 40;
                auto wrapped = wrap_plain(c, wrap_w);
                for (size_t k = 0; k < wrapped.size(); ++k) {
                    if (k == 0) std::cout << C_QUOTE << "  \xe2\x94\x83 ";
                    else        std::cout << "    ";
                    std::cout << render_inline_md(wrapped[k]) << C_RESET << "\n";
                }
            }
            continue;
        }

        // Горизонтальная линия
        if (line.size()>=3) {
            bool hr = true; char ch = line[0];
            if (ch=='-'||ch=='*'||ch=='_') {
                for (char x:line) if(x!=ch&&x!=' '){hr=false;break;}
            } else hr=false;
            if (hr) {
                std::cout << C_GRAY
                    << "  \xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
                       "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
                       "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
                       "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
                       "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
                       "\xe2\x94\x80\xe2\x94\x80"
                    << C_RESET << "\n";
                continue;
            }
        }

        // Маркированный список (- или *)
        if (line.size()>=2 && (line[0]=='-'||line[0]=='*') && line[1]==' ') {
            {
                std::string body = line.substr(2);
                if (is_compact()) {
                    std::cout << "  \xe2\x80\xa2 " << body << "\n";
                } else {
                    int tw = get_terminal_width();
                    size_t wrap_w = (tw > 12) ? static_cast<size_t>((tw - 4)) : 40;
                    auto wrapped = wrap_plain(body, wrap_w);
                    for (size_t k = 0; k < wrapped.size(); ++k) {
                        if (k == 0) std::cout << C_BULLET << "  \xe2\x80\xa2 " << C_RESET;
                        else        std::cout << "    ";
                        std::cout << render_inline_md(wrapped[k]) << "\n";
                    }
                }
            }
            continue;
        }

        // Нумерованный список
        { size_t p=0;
          while(p<line.size()&&line[p]>='0'&&line[p]<='9') ++p;
          if (p>0&&p<line.size()&&line[p]=='.'&&p+1<line.size()&&line[p+1]==' ') {
              {
                  std::string num = line.substr(0, p);
                  std::string body = line.substr(p + 2);
                  std::string prefix = "  " + num + ". ";
                  if (is_compact()) {
                      std::cout << prefix << body << "\n";
                  } else {
                      int tw = get_terminal_width();
                      size_t prefix_w = 2 + num.size() + 2;
                      size_t wrap_w = (tw > static_cast<int>(prefix_w) + 8) ? static_cast<size_t>((tw - prefix_w)) : 40;
                      auto wrapped = wrap_plain(body, wrap_w);
                      for (size_t k = 0; k < wrapped.size(); ++k) {
                          if (k == 0) std::cout << C_BULLET << prefix << C_RESET;
                          else        std::cout << std::string(prefix_w, ' ');
                          std::cout << render_inline_md(wrapped[k]) << "\n";
                      }
                  }
              }
              continue;
          }
        }

        // Вложенный список (2+ пробела)
        if (line.size()>=4) {
            size_t sp=0;
            while(sp<line.size()&&line[sp]==' ') ++sp;
            if (sp>=2 && sp<line.size() && (line[sp]=='-'||line[sp]=='*')
                && sp+1<line.size() && line[sp+1]==' ') {
                std::string ind(sp/2, ' ');
                std::string body = line.substr(sp + 2);
                std::string prefix = "  " + ind + "\xe2\x97\xa6 ";
                if (is_compact()) {
                    std::cout << prefix << body << "\n";
                } else {
                    int tw = get_terminal_width();
                    size_t prefix_w = 2 + ind.size() + 3; // "  " + ind + "◦ "
                    size_t wrap_w = (tw > static_cast<int>(prefix_w) + 8) ? static_cast<size_t>((tw - prefix_w)) : 40;
                    auto wrapped = wrap_plain(body, wrap_w);
                    for (size_t k = 0; k < wrapped.size(); ++k) {
                        if (k == 0) std::cout << C_BULLET << prefix << C_RESET;
                        else        std::cout << std::string(prefix_w, ' ');
                        std::cout << render_inline_md(wrapped[k]) << "\n";
                    }
                }
                continue;
            }
        }

        // ── Markdown таблицы (двухпроходный рендер) ──
        std::vector<size_t> table_col_widths;
        if (line.find('|') != std::string::npos) {
            auto first_cells = split_table_cells(replace_flags(line));
            if (first_cells.size() >= 2) {
                auto check_sep = [](const std::string &l) -> bool {
                    bool has_dash = false;
                    for (char c : l) {
                        if (c == '-') has_dash = true;
                        else if (c != '|' && c != ' ' && c != ':') return false;
                    }
                    return has_dash;
                };
                std::vector<std::vector<std::string>> table_rows;
                std::vector<bool> is_sep_row;
                table_rows.push_back(first_cells);
                is_sep_row.push_back(check_sep(line));
                std::string next_line;
                bool has_leftover = false;
                std::string leftover;
                while (std::getline(ss, next_line)) {
                    if (!next_line.empty() && next_line.back() == '\r') next_line.pop_back();
                    if (next_line.find('|') == std::string::npos) {
                        has_leftover = true; leftover = next_line; break;
                    }
                    auto nc = split_table_cells(replace_flags(next_line));
                    if (nc.size() < 2) { has_leftover = true; leftover = next_line; break; }
                    is_sep_row.push_back(check_sep(next_line));
                    table_rows.push_back(nc);
                }
                size_t max_cols = 0;
                for (auto &row : table_rows) if (row.size() > max_cols) max_cols = row.size();
                table_col_widths.assign(max_cols, 0);
                for (size_t ri = 0; ri < table_rows.size(); ++ri) {
                    if (is_sep_row[ri]) continue;
                    for (size_t ci = 0; ci < table_rows[ri].size(); ++ci) {
                        size_t w = visible_width(render_inline_md(table_rows[ri][ci]));
                        if (w > table_col_widths[ci]) table_col_widths[ci] = w;
                    }
                }
                for (auto &w : table_col_widths) if (w < 3) w = 3;
                // Адаптация к ширине терминала (и в компакт-режиме тоже)
                {
                    int tw = get_terminal_width();
                    if (tw > 20) {
                        // Точная ширина таблицы: 1 + Σ(col_w + 3) = 1 + Σ(col_w) + 3*max_cols
                        auto table_total = [&]() -> size_t {
                            size_t t = 1 + 3 * max_cols;
                            for (auto w : table_col_widths) t += w;
                            return t;
                        };
                        // Минимальная ширина колонки: 3 обычно, 2 — если иначе не влезаем
                        size_t min_col = 3;
                        if (1 + 3 * max_cols + min_col * max_cols > static_cast<size_t>(tw))
                            min_col = 2;
                        // Сжимаем самую широкую колонку, пока таблица не влезет
                        while (table_total() > static_cast<size_t>(tw)) {
                            size_t wi = 0;
                            for (size_t ci = 1; ci < table_col_widths.size(); ++ci)
                                if (table_col_widths[ci] > table_col_widths[wi]) wi = ci;
                            if (table_col_widths[wi] <= min_col) break;
                            --table_col_widths[wi];
                        }
                    }
                }
                std::cout << C_GRAY << "\xe2\x94\x8c";
                for (size_t ci = 0; ci < max_cols; ++ci) {
                    for (size_t k = 0; k < table_col_widths[ci] + 2; ++k) std::cout << "\xe2\x94\x80";
                    std::cout << ((ci + 1 < max_cols) ? "\xe2\x94\xac" : "\xe2\x94\x90");
                }
                std::cout << C_RESET << "\n";
                bool header_done = false;
                for (size_t ri = 0; ri < table_rows.size(); ++ri) {
                    if (is_sep_row[ri]) {
                        std::cout << C_GRAY << "\xe2\x94\x9c";
                        for (size_t ci = 0; ci < max_cols; ++ci) {
                            for (size_t k = 0; k < table_col_widths[ci] + 2; ++k) std::cout << "\xe2\x94\x80";
                            std::cout << ((ci + 1 < max_cols) ? "\xe2\x94\xbc" : "\xe2\x94\xa4");
                        }
                        std::cout << C_RESET << "\n";
                        header_done = true;
                        continue;
                    }
                    render_table_row(table_rows[ri], table_col_widths, !header_done);
                }
                std::cout << C_GRAY << "\xe2\x94\x94";
                for (size_t ci = 0; ci < max_cols; ++ci) {
                    for (size_t k = 0; k < table_col_widths[ci] + 2; ++k) std::cout << "\xe2\x94\x80";
                    std::cout << ((ci + 1 < max_cols) ? "\xe2\x94\xb4" : "\xe2\x94\x98");
                }
                std::cout << C_RESET << "\n";
                if (has_leftover && !leftover.empty()) {
                    std::cout << render_inline_md(leftover) << "\n";
                }
                continue;
            }
        }
        
        if (is_compact()) {
            // В компакт-режиме тоже раскрываем инлайн-разметку (**жирный**, `код`),
            // но БЕЗ переноса строк (compact = без word-wrap).
            std::cout << render_inline_md(line) << C_RESET << "\n";
        } else {
            int tw = get_terminal_width();
            auto wrapped = wrap_plain(line, static_cast<size_t>((tw > 8 ? tw : 80)));
            for (auto &wseg : wrapped)
                std::cout << render_inline_md(wseg) << "\n";
        }
    }
    if (in_code) {
        for (auto &cl : realign_box_art(code_lines))
            std::cout << C_GRAY << "\xe2\x94\x82 " << C_WHITE << cl << C_RESET << "\n";
        code_lines.clear();
        std::cout << C_GRAY
                  << "\xe2\x94\x94\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
                     "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
                     "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
                  << C_RESET << "\n";
    }
}

// ─────────────────────────── История ─────────────────────────
void save_history(bool silent = false) {
    try {
        json j = json::array();
        for (auto &m : G.messages) j.push_back(m);
        std::ofstream f(G.history_file);
        if (f.is_open()) {
            f << j.dump(2, ' ', false, json::error_handler_t::replace);
            if (!silent) std::cout << C_YELLOW << "[История сохранена: " << G.history_file
                      << "]" << C_RESET << std::endl;
        } else {
            std::cerr << C_RED << "[Не удалось открыть файл истории для записи]"
                      << C_RESET << std::endl;
        }
    } catch (...) {
        std::cerr << C_RED << "[Ошибка сохранения истории]" << C_RESET << std::endl;
    }
}

bool load_history() {
    std::ifstream f(G.history_file);
    if (!f.is_open()) return false;
    try {
        std::string content((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
        if (content.empty()) return false;
        json j = json::parse(content);
        if (!j.is_array()) return false;
        std::vector<json> loaded;
        for (auto &m : j) {
            if (m.is_object()) loaded.push_back(m);
        }
        G.messages.swap(loaded);
        if (G.messages.empty() || G.messages[0].value("role", "") != "system") {
            G.messages.insert(G.messages.begin(),
                {{"role", "system"}, {"content", G.sys_prompt}});
        }
        std::cout << C_YELLOW << "[История загружена: " << G.messages.size()
                  << " сообщений]" << C_RESET << std::endl;
        return true;
    } catch (const std::exception &e) {
        std::cerr << C_RED << "[Ошибка загрузки истории: " << e.what() << "]" << C_RESET << std::endl;
        return false;
    }
}

std::string load_system_prompt() {
    std::ifstream f(SYSTEM_PROMPT_FILE);
    if (!f.is_open()) return "";
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    if (!G.fire)
        std::cout << C_YELLOW << "[Системный промпт загружен из "
                  << SYSTEM_PROMPT_FILE << "]" << C_RESET << std::endl;
    return content;
}

// ─────────────────────────── Обрезка контекста ───────────────
// compact_mode quiet UX helpers (early: only need GlobalState G)
static bool is_compact() { return G.compact_mode; }
static void note_gray(const std::string& s) {
    if (is_compact()) return;
    std::cout << C_GRAY << s << C_RESET << std::endl;
}
static void note_yellow(const std::string& s) {
    if (is_compact()) return;
    std::cout << C_YELLOW << s << C_RESET << std::endl;
}

static void print_assistant_text(const std::string& content, bool with_header = true) {
    if (content.empty()) return;
    if (G.fire || is_compact()) {
        // FIRE и compact: без заголовка [Ассистент], но через render_markdown —
        // так markdown-таблицы адаптируются под ширину экрана.
        // render_markdown в compact печатает обычный текст как есть.
        render_markdown(content);
        return;
    }
    if (with_header)
        std::cout << "\n" << C_BOLD << C_CYAN << "[Ассистент]:" << C_RESET << "\n";
    render_markdown(content);
    std::cout << std::endl;
}

void trim_messages_if_needed() {
    if (static_cast<int>(G.messages.size()) <= MAX_MESSAGES) return;

    std::vector<json> trimmed;
    int sys_idx = -1;
    for (int i = 0; i < static_cast<int>(G.messages.size()); ++i) {
        if (G.messages[i].value("role", "") == "system") { sys_idx = i; break; }
    }
    if (sys_idx >= 0) trimmed.push_back(G.messages[sys_idx]);

    int keep_count = MAX_MESSAGES - (sys_idx >= 0 ? 1 : 0);
    int start_from = static_cast<int>(G.messages.size()) - keep_count;
    if (start_from < 0) start_from = 0;

    for (int i = start_from; i < static_cast<int>(G.messages.size()); ++i) {
        if (i == sys_idx) continue;
        trimmed.push_back(G.messages[i]);
    }
    G.messages = trimmed;
    if (!is_compact()) {
        std::cout << C_GRAY << "[Контекст обрезан до " << G.messages.size()
                  << " сообщений]" << C_RESET << std::endl;
    }
}

// ─────────────────────────── Shell exec ──────────────────────
// Последний вывод bash (для /dump)
static std::string g_last_bash_result;
static std::string g_last_bash_code;

static std::string shell_escape(const std::string& s) {
    std::string result = "'";
    for (char c : s) {
        if (c == '\'') result += "'\\''";
        else result += c;
    }
    result += "'";
    return result;
}

struct ExecResult {
    std::string output;
    int exit_code = -1;
    bool timed_out = false;
    bool popen_failed = false;
    double duration_sec = 0.0;   // реальное время выполнения (сек)
};

static std::string format_exec_result(const ExecResult& er, int timeout_sec) {
    std::string body = er.output;
    while (!body.empty() && (body.back() == '\n' || body.back() == '\r' || body.back() == ' '))
        body.pop_back();
    std::string out;
    if (er.popen_failed) {
        out = "[popen failed]";
    } else if (body.empty()) {
        // A4: explicit empty output so the model does not assume failure/silence
        out = "(empty output, exit " + std::to_string(er.exit_code) + ")";
    } else {
        out = er.output;
        if (!out.empty() && out.back() != '\n') out.push_back('\n');
        // A5: always expose exit code to the model / user context
        out += "[exit: " + std::to_string(er.exit_code) + "]";
    }
    if (er.timed_out)
        out += "\n[ТАЙМАУТ: команда прервана после " + std::to_string(timeout_sec) + " сек]";
    return out;
}

ExecResult exec_with_timeout_ex(const std::string& cmd, int timeout_sec) {
    ExecResult er;
    int pipefd[2];
    if (pipe(pipefd) != 0) { er.popen_failed = true; er.exit_code = 127; return er; }

    pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]); close(pipefd[1]);
        er.popen_failed = true; er.exit_code = 127; return er;
    }

    if (pid == 0) {
        // Дочерний процесс: новая сессия/группа процессов, чтобы можно было
        // убить всё дерево одним kill(-pid). stdout+stderr → pipe.
        setsid();
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        if (pipefd[1] > STDERR_FILENO) close(pipefd[1]);
        execl("/system/bin/sh", "sh", "-c", cmd.c_str(), static_cast<char*>(nullptr));
        execl("/bin/sh", "sh", "-c", cmd.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }

    // Родитель: читаем с poll() короткими слайсами — гарантирует
    // реакцию на Ctrl+C (g_stream_abort) в течение ~200 мс.
    close(pipefd[1]);
    int fd = pipefd[0];
    std::string outp;
    char buf[4096];
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);
    bool timed_out = false, interrupted = false, child_done = false;

    while (true) {
        if (g_stream_abort) { interrupted = true; break; }
        struct pollfd pfd;
        pfd.fd = fd; pfd.events = POLLIN; pfd.revents = 0;
        int pr = poll(&pfd, 1, 200);
        if (pr < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (pr == 0) {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            double elapsed = static_cast<double>((now.tv_sec - start.tv_sec)) +
                             static_cast<double>((now.tv_nsec - start.tv_nsec)) / 1e9;
            if (elapsed >= static_cast<double>(timeout_sec)) { timed_out = true; break; }
            continue;
        }
        ssize_t r = read(fd, buf, sizeof(buf));
        if (r < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (r == 0) { child_done = true; break; }
        outp.append(buf, static_cast<size_t>(r));
        if (outp.size() > static_cast<size_t>(MAX_CMD_OUTPUT) * 4) break; // защита от ^C-спама
    }

    // Прибираем дочернее дерево, если оно ещё живо.
    int status = 0;
    if (!child_done) {
        kill(-pid, SIGTERM);
        for (int i = 0; i < 10; ++i) {
            pid_t w = waitpid(pid, &status, WNOHANG);
            if (w == pid) { child_done = true; break; }
            if (w < 0 && errno == ECHILD) { child_done = true; break; }
            struct timespec ts; ts.tv_sec = 0; ts.tv_nsec = 50L*1000*1000;
            nanosleep(&ts, nullptr);
        }
        if (!child_done) {
            kill(-pid, SIGKILL);
        }
    }
    // добираем вывод, что остался в буфере pipe
    while (true) {
        ssize_t r = read(fd, buf, sizeof(buf));
        if (r <= 0) break;
        outp.append(buf, static_cast<size_t>(r));
        if (outp.size() > static_cast<size_t>(MAX_CMD_OUTPUT) * 4) break;
    }
    // FIX(BUG1): всегда дожидаемся завершения ребёнка, чтобы получить
    // корректный status. Раньше при нормальном выходе (read()==0 ->
    // child_done=true) этот waitpid пропускался и exit_code всегда был 0.
    for (;;) {
        pid_t w = waitpid(pid, &status, 0);
        if (w == pid) break;
        if (w < 0 && errno == EINTR) continue;
        break; // ECHILD (уже убран) или другая ошибка
    }
    close(fd);

    if (WIFEXITED(status)) {
        er.exit_code = WEXITSTATUS(status);
        if (er.exit_code == 124) er.timed_out = true;
    } else if (WIFSIGNALED(status)) {
        er.exit_code = 128 + WTERMSIG(status);
    } else {
        er.exit_code = -1;
    }
    er.timed_out = er.timed_out || timed_out;
    if (interrupted) er.exit_code = 130; // 128 + SIGINT

    {
        struct timespec end; clock_gettime(CLOCK_MONOTONIC, &end);
        er.duration_sec = static_cast<double>((end.tv_sec - start.tv_sec)) +
                          static_cast<double>((end.tv_nsec - start.tv_nsec)) / 1e9;
        if (er.duration_sec < 0) er.duration_sec = 0;
    }

    if (outp.size() > static_cast<size_t>(MAX_CMD_OUTPUT)) {
        size_t cut = MAX_CMD_OUTPUT;
        while (cut > 0 && (outp[cut] & 0xC0) == 0x80) --cut;
        outp = outp.substr(0, cut) +
               "\n[...вывод обрезан, превышен лимит " +
               std::to_string(MAX_CMD_OUTPUT) + " байт...]";
    }
    er.output = outp;
    if (interrupted)
        er.output += "\n[ПРЕРВАНО пользователем (Ctrl+C)]";
    return er;
}

// back-compat wrapper
std::string exec_with_timeout(const std::string& cmd, int timeout_sec) {
    return format_exec_result(exec_with_timeout_ex(cmd, timeout_sec), timeout_sec);
}

// ─────────────────────────── Голос (Termux API) ──────────────
static bool have_termux_api(const std::string& cmd) {
    const char* pfx = getenv("PREFIX");
    std::string base = pfx ? std::string(pfx) : std::string("/usr");
    return access((base + "/bin/" + cmd).c_str(), X_OK) == 0;
}

// Речь → текст (termux-speech-to-text отдаёт JSON {"text":"..."}).
static std::string voice_recognize() {
    if (!have_termux_api("termux-speech-to-text")) return "";
    g_stream_abort = 0;
    g_in_bash = 1;   // Ctrl+C прерывает распознавание, не программу
    ExecResult er = exec_with_timeout_ex("termux-speech-to-text", 30);
    g_in_bash = 0;
    g_stream_abort = 0;
    if (er.timed_out || er.exit_code != 0) return "";
    std::string out = er.output;
    while (!out.empty() && (out.back()=='\n'||out.back()=='\r'||out.back()==' ')) out.pop_back();
    if (out.empty() || out[0] != '{') return "";
    try {
        auto j = json::parse(out);
        if (j.is_object() && j.count("text") && j["text"].is_string())
            return sanitize_utf8(j["text"].get<std::string>());
    } catch (...) {}
    return "";
}

// Очистка markdown/кода для озвучки.
static std::string strip_for_tts(const std::string& s, bool first_paragraph_only = false) {
    std::string out;
    out.reserve(s.size());
    std::istringstream ss(s);
    std::string line;
    bool in_code = false;
    while (std::getline(ss, line)) {
        if (!line.empty() && line.back()=='\r') line.pop_back();
        if (line.size() >= 3 && line.substr(0,3) == "```") { in_code = !in_code; continue; }
        if (in_code) continue;
        if (line.empty()) {
            if (first_paragraph_only && !out.empty()) break;
            continue;
        }
        if (line[0]=='#' || line[0]=='|' || line[0]=='>') continue;
        size_t st = 0;
        while (st < line.size() && (line[st]==' '||line[st]=='\t')) ++st;
        if (st+1 < line.size() && (line[st]=='-'||line[st]=='*') && line[st+1]==' ') st += 2;
        {
            size_t q = st;
            while (q < line.size() && line[q]>='0' && line[q]<='9') ++q;
            if (q > st && q+1 < line.size() && line[q]=='.' && line[q+1]==' ') st = q+2;
        }
        std::string clean;
        for (size_t i = st; i < line.size(); ++i) {
            char c = line[i];
            if (c=='`'||c=='*'||c=='_'||c=='#'||c=='['||c==']'||c=='('||c==')') continue;
            clean += c;
        }
        while (!clean.empty() && clean.back()==' ') clean.pop_back();
        if (clean.empty()) continue;
        out += clean;
        out += ". ";
    }
    const size_t MAX_TTS = 800;
    if (out.size() > MAX_TTS) {
        size_t cut = MAX_TTS;
        while (cut>0 && (static_cast<unsigned char>(out[cut])&0xC0)==0x80) --cut;
        out = out.substr(0, cut) + "...";
    }
    return out;
}

// Озвучка в фоне (не блокирует интерфейс).
static void voice_speak(const std::string& text) {
    if (text.empty()) return;
    if (!have_termux_api("termux-tts-speak")) return;
    std::string cmd = "termux-tts-speak";
    if (!G.voice_engine.empty())  cmd += " -e " + shell_escape(G.voice_engine);
    if (!G.voice_lang.empty())    cmd += " -l " + shell_escape(G.voice_lang);
    if (!G.voice_region.empty())  cmd += " -n " + shell_escape(G.voice_region);
    if (!G.voice_variant.empty()) cmd += " -v " + shell_escape(G.voice_variant);
    if (G.voice_pitch != 1.0) { char b[32]; snprintf(b, sizeof(b), " -p %.2f", G.voice_pitch); cmd += b; }
    if (G.voice_rate  != 1.0) { char b[32]; snprintf(b, sizeof(b), " -r %.2f", G.voice_rate);  cmd += b; }
    if (!G.voice_stream.empty())  cmd += " -s " + shell_escape(G.voice_stream);
    cmd += " " + shell_escape(text) + " </dev/null >/dev/null 2>&1 &";
    system(cmd.c_str());
}

// Выполняет один bash-блок с подтверждением
// FIRE: эвристика потенциально опасных команд (спрашиваем даже в FIRE).
static bool looks_dangerous(const std::string& cmd) {
    // P0.4: расширенная эвристика. Это НЕ защита, а вежливое "точно ли?".
    // Канонизируем вход:
    //   - lowercase;
    //   - переносы строк/табы -> пробел (одномерная форма);
    //   - сжатие подряд идущих пробелов;
    //   - выкидываем кавычки: обход "rm -rf $HOME" -> "rm -rf ~";
    //   - $HOME / ${HOME} -> ~ (чтобы стандартные варианты ловились).
    std::string c;
    c.reserve(cmd.size());
    bool sp = false;
    for (char ch : cmd) {
        if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {
            if (!sp) c += ' ';
            sp = true;
        } else if (ch == '"' || ch == '\'') {
            continue;   // кавычки выкидываем
        } else {
            c += static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            sp = false;
        }
    }
    while (!c.empty() && c.back() == ' ') c.pop_back();

    auto replace_all = [](std::string& s, const std::string& from, const std::string& to) {
        if (from.empty()) return;
        size_t pos = 0;
        while ((pos = s.find(from, pos)) != std::string::npos) {
            s.replace(pos, from.size(), to);
            pos += to.size();
        }
    };
    replace_all(c, "${home}", "~");
    replace_all(c, "$home",   "~");

    static const char* pats[] = {
        // перезапись/удаление корня и домашнего каталога
        "rm -rf /", "rm -fr /",
        "rm -rf /*", "rm -fr /*",
        "rm -rf ~", "rm -fr ~",
        "rm -rf ~/*", "rm -fr ~/*",
        "rm -rf --no-preserve-root", "rm -fr --no-preserve-root",
        "rm -rf .", "rm -fr .",           // снести текущий каталог
        // fork bomb
        ":(){", ": (){", ":() {",
        // дисковые и низкоуровневые
        "mkfs", "wipefs", "shred", "blkdiscard",
        "of=/dev/", "of=/dev",
        "> /dev/sd", "> /dev/nvme", "> /dev/mmcblk",
        "> /dev/hd", "> /dev/da", "> /dev/vd", "> /dev/xvd",
        // системные каталоги
        "> /etc/", "> /boot/", "> /sys/", "> /proc/",
        // опасные права
        "chmod -r 777 /", "chmod 777 -r /",
        "chmod -r 000 /",
        "chown -r /", "chown -r /*",
        // пайп в шелл (выполнить чужой код без просмотра)
        "| sh", "|sh", "| bash", "|bash",
        // массовое удаление через find
        "find / -delete", "find / -exec rm",
        // питание/инициализация
        "shutdown", "reboot", "halt", "poweroff", "init 0", "init 6",
        // глобальный kill
        "kill -9 -1", "killall5",
        // eval динамической строки
        "eval $(", "eval `",
    };
    // dd опасен только в связке с of=
    if (c.find("dd ") != std::string::npos && c.find("of=") != std::string::npos) return true;
    for (const char* p : pats) if (c.find(p) != std::string::npos) return true;
    return false;
}

// local_autorun — флаг "запустить без вопросов до конца цепочки":
// действует на все блоки текущего пакета И на последующие ответы модели
// в рамках одной bash-цепочки. Не трогает G.autorun (это отдельный глобальный флаг).
std::string execute_single_bash(const std::string &bash_code, int idx, int total, bool &local_autorun) {
    if (total > 1 && !G.fire)
        std::cout << C_YELLOW << "[Bash блок " << (idx+1) << "/" << total << "]" << C_RESET << std::endl;
    // ── DRY-RUN: не выполняем, только показываем ──
    if (g_dry_run) {
        std::cout << C_CYAN << "[DRY-RUN: команда не выполнена]" << C_RESET << std::endl;
        if (!is_compact()) {
            std::cout << C_GRAY << "--- команда ---" << C_RESET << std::endl;
            std::cout << C_CODE_FG << bash_code << C_RESET << std::endl;
            std::cout << C_GRAY << "--- конец ---" << C_RESET << std::endl;
        }
        std::string preview = "[DRY-RUN: команда НЕ выполнена. Блок " +
                              std::to_string(idx+1) + "/" + std::to_string(total) +
                              "]\n```bash\n" + bash_code + "\n```";
        g_last_bash_result = preview;
        g_last_bash_code   = bash_code;
        return preview;
    }

    // FIRE: выполняем молча; опасные команды — исключение.
    if (G.fire) {
        if (looks_dangerous(bash_code)) {
            std::cout << C_RED << C_BOLD
                      << "⚠ FIRE: похоже на опасную команду. Всё равно выполнить? "
                      << C_RESET << std::flush;
            char* rl = readline(C_RED "(y/n) " C_RESET);
            std::string ans = rl ? std::string(rl) : std::string();
            if (rl) free(rl);
            if (!(ans == "y" || ans == "Y" || ans == "д" || ans == "Д")) {
                std::cout << C_RED << "[FIRE: опасная команда отклонена]" << C_RESET << std::endl;
                return "[Пользователь отказался выполнять эту команду]";
            }
        }
    } else if (!G.autorun && !local_autorun) {
        // 'a' (выполнить без вопросов до конца цепочки) предлагаем только когда
        // в текущем ответе блоков больше одного — иначе это лишний вариант.
        // Если 'a' нажато, флаг local_autorun живёт до конца всей bash-цепочки
        // (объявлен в process_response), а не только текущего ответа.
        const char* prompt;
        if (is_compact()) {
            // Короткий режим: только минимальная подсказка. Русские д/н/в
            // всё равно принимаются обработчиком ниже.
            prompt = (total > 1) ? C_YELLOW "(y/n/a) " C_RESET
                                 : C_YELLOW "(y/n) " C_RESET;
        } else {
            prompt = (total > 1)
                ? C_YELLOW "[Выполнить команду? (y/n/a-все|д/н/в)]: " C_RESET
                : C_YELLOW "[Выполнить команду? (y/n | д/н)]: " C_RESET;
        }
        char *rl = readline(prompt);
        if (!rl) return "[Пользователь отказался выполнять эту команду]";
        std::string ans(rl); free(rl);
        if (total > 1 && (ans == "a" || ans == "A" || ans == "в" || ans == "В")) {
            local_autorun = true;
        } else if (ans != "y" && ans != "Y" && ans != "д" && ans != "Д") {
            if (!is_compact())
                std::cout << C_RED << "[Блок " << (idx+1) << " пропущен]" << C_RESET << std::endl;
            return "[Пользователь отказался выполнять эту команду]";
        }
    } else if (!G.fire && !is_compact()) {
        if (total > 1)
            std::cout << C_YELLOW << "[Autorun: выполняю блок " << (idx+1) << "]" << C_RESET << std::endl;
        else
            std::cout << C_YELLOW << "[Autorun: выполняю автоматически]" << C_RESET << std::endl;
        std::cout << C_YELLOW << "[Выполняю...]" << C_RESET << std::endl;
    }
    // Помечаем, что идёт bash: Ctrl+C прервёт команду, а не всю программу.
    g_stream_abort = 0;
    g_in_bash = 1;
    ExecResult er = exec_with_timeout_ex(bash_code, CMD_TIMEOUT);
    g_in_bash = 0;
    g_stream_abort = 0;
    std::string result = format_exec_result(er, CMD_TIMEOUT);
    g_last_bash_result = result;
    g_last_bash_code = bash_code;

    // ── Всегда показываем результат/статус (даже при nores) ──
    // Строка "тела" без служебных хвостов [exit: N] и таймаута.
    std::string body = er.output;
    while (!body.empty() && (body.back() == '\n' || body.back() == '\r' || body.back() == ' '))
        body.pop_back();
    bool empty_out = body.empty();
    bool interrupted = (er.exit_code == 130);

    // Цвет статуса
    const char* stat_color = interrupted ? C_YELLOW
                           : (er.exit_code == 0 ? C_GREEN : C_RED);

    if (G.fire) {
        // FIRE: ни строчки про bash (код/результат/статус)
    } else if (is_compact()) {
        if (!G.nores && !result.empty()) {
            // Вывод команды видим — печатаем как есть (сам вывод = обратная связь).
            std::cout << result;
            if (result.back() != char(10)) std::cout << char(10);
        } else if (interrupted) {
            // Молчим на штатный успех — сообщаем только о проблемах.
            std::cout << C_YELLOW << "[прервано]" << C_RESET << std::endl;
        } else if (er.timed_out) {
            std::cout << C_YELLOW << "[таймаут]" << C_RESET << std::endl;
        } else if (er.exit_code != 0) {
            std::cout << C_RED << "[exit " << er.exit_code << "]" << C_RESET << std::endl;
        }
        // exit 0 без таймаута/прерывания — ничего не печатаем (шум не нужен).
    } else {
        // Полный (некомпактный) режим: результат + подробная статусная строка.
        if (!G.nores) {
            if (empty_out && !interrupted && !er.timed_out && er.exit_code == 0)
                std::cout << C_BLUE << "[Результат]: (пусто)" << C_RESET << std::endl;
            else
                std::cout << C_BLUE << "[Результат]:\n" << result << C_RESET << std::endl;
        }
        char dur[32];
        snprintf(dur, sizeof(dur), "%.2fс", er.duration_sec);
        std::string state;
        if (interrupted)          state = "прервано";
        else if (er.timed_out)    state = "таймаут";
        else if (empty_out)       state = "нет вывода";
        else                      state = std::to_string(body.size()) + " байт";
        std::cout << C_GRAY << "[готово: exit " << C_RESET
                  << stat_color << er.exit_code << C_RESET
                  << C_GRAY << " · " << state << " · " << dur << "]"
                  << C_RESET << std::endl;
    }
    return result;
}


// ─────────────────────────── Спиннер ─────────────────────────
// Thread-safe: поток спиннера пишет ТОЛЬКО в stderr через write() (async-safe),
// основной поток во время curl_easy_perform в stdout/stderr не пишет.
static std::atomic<bool> g_spinner_run{false};

static void spinner_loop(std::string model) {
    static const char* frames[] = {
        "\xe2\xa0\x8b","\xe2\xa0\x99","\xe2\xa0\xb9","\xe2\xa0\xb8",
        "\xe2\xa0\xbc","\xe2\xa0\xb4","\xe2\xa0\xa6","\xe2\xa0\xa7",
        "\xe2\xa0\x87","\xe2\xa0\x8f"
    };
    int idx = 0;
    while (g_spinner_run.load(std::memory_order_relaxed)) {
        std::string line = std::string("\r") + C_CYAN + frames[idx % 10] +
                           C_RESET + C_GRAY + " размышляю " + C_YELLOW +
                           "[" + model + "]" + C_RESET + "\033[K";
        ssize_t r = ::write(STDERR_FILENO, line.data(), line.size());
        (void)r;
        idx++;
        std::this_thread::sleep_for(std::chrono::milliseconds(90));
    }
    const char* clr = "\r\033[K";
    ssize_t r = ::write(STDERR_FILENO, clr, 5); (void)r;
}

static std::thread g_spinner_thread;
static void spinner_start(const std::string &model) {
    if (is_compact()) return; // quiet
    if (g_spinner_run.load()) return;
    g_spinner_run.store(true);
    g_spinner_thread = std::thread(spinner_loop, model);
}
static void spinner_stop() {
    if (!g_spinner_run.load()) return;
    g_spinner_run.store(false);
    if (g_spinner_thread.joinable()) g_spinner_thread.join();
}

// ─────────────────────────── API запрос ──────────────────────
// Возвращает full_content или "" при ошибке/прерывании
// aborted — true если пользователь прервал Ctrl+C
// Простой callback для накопления ответа
static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    if (g_stream_abort) return 0;
    std::string *buf = static_cast<std::string*>(userp);
    buf->append(static_cast<char*>(contents), size * nmemb);
    return size * nmemb;
}

// Progress-колбэк libcurl: вызывается регулярно (в т.ч. в фазе ожидания
// соединения/первого байта). Возврат != 0 немедленно прерывает transfer.
// Это основной способ отреагировать на Ctrl+C до прихода данных.
static int curl_progress_cb(void* /*clientp*/, curl_off_t /*dltotal*/, curl_off_t /*dlnow*/,
                            curl_off_t /*ultotal*/, curl_off_t /*ulnow*/) {
    return g_stream_abort ? 1 : 0;
}

// ──────────────────── Директории и конфиг ────────────────────
// Рекурсивно создаёт директорию вместе с родителями (аналог `mkdir -p`).
// Нужно для портируемости: на Linux ~/.config может отсутствовать.
static void ensure_dir(const std::string& path) {
    if (path.empty()) return;
    struct stat st;
    if (stat(path.c_str(), &st) == 0) return;   // уже есть
    std::string parent = path;
    size_t slash = parent.find_last_of('/');
    if (slash != std::string::npos && slash > 0) {
        parent = parent.substr(0, slash);
        struct stat pst;
        if (stat(parent.c_str(), &pst) != 0) ensure_dir(parent);  // создать родителя
    }
    mkdir(path.c_str(), 0755);
}
static void init_paths() {
    std::string home = get_home_dir();
    CONFIG_DIR     = home + "/.config/sw_chat";
    CONFIG_FILE    = CONFIG_DIR + "/config.json";
    SESSIONS_DIR   = CONFIG_DIR + "/sessions";
    HISTORY_FILE   = SESSIONS_DIR + "/" + G.session_name + ".json";
    READLINE_HIST_FILE = CONFIG_DIR + "/.readline_history";
    MODELS_CACHE_FILE = CONFIG_DIR + "/models.json";
    ensure_dir(CONFIG_DIR); ensure_dir(SESSIONS_DIR);
    G.history_file = HISTORY_FILE;
}
static std::string normalize_api_base(std::string u) {
    while (!u.empty() && (u.back() == '/' || u.back() == ' ')) u.pop_back();
    if (u.empty()) u = DEFAULT_API_BASE;
    return u;
}
static void save_config() {
    try {
        json j; j["model"]=G.model; j["temperature"]=G.temperature; j["max_tokens"]=G.max_tokens;
        j["autorun"]=G.autorun; j["history_enabled"]=G.history_enabled; j["nores"]=G.nores;
        j["compact_mode"]=G.compact_mode;
        j["fire"]=G.fire;
        j["time_prefix"]=G.time_prefix;
        j["voice_in"]=G.voice_in; j["voice_out"]=G.voice_out;
        j["voice_lang"]=G.voice_lang; j["voice_region"]=G.voice_region;
        j["voice_variant"]=G.voice_variant; j["voice_engine"]=G.voice_engine;
        j["voice_stream"]=G.voice_stream; j["voice_pitch"]=G.voice_pitch;
        j["voice_rate"]=G.voice_rate; j["voice_short"]=G.voice_short;
        j["api_base"]=G.api_base;
        j["aliases"]=G.aliases;
        j["session_name"]=G.session_name;
        std::ofstream f(CONFIG_FILE); if(f.is_open()) f << j.dump(2);
    } catch(...){}
}
static void load_config() {
    std::ifstream f(CONFIG_FILE); if(!f.is_open()) return;
    try {
        std::string c((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if(c.empty()) return; json j = json::parse(c);
        if(j.count("model")) G.model=j["model"]; if(j.count("temperature")) G.temperature=j["temperature"];
        if(j.count("max_tokens")) G.max_tokens=j["max_tokens"]; if(j.count("autorun")) G.autorun=j["autorun"];
        if(j.count("history_enabled")) G.history_enabled=j["history_enabled"]; if(j.count("nores")) G.nores=j["nores"];
        if(j.count("compact_mode")) G.compact_mode=j["compact_mode"];
        if(j.count("fire")) G.fire=j["fire"];
        if(j.count("time_prefix")) G.time_prefix=j["time_prefix"];
        if(j.count("voice_in")) G.voice_in=j["voice_in"];
        if(j.count("voice_out")) G.voice_out=j["voice_out"];
        if(j.count("voice_lang")    && j["voice_lang"].is_string())    G.voice_lang=j["voice_lang"];
        if(j.count("voice_region")  && j["voice_region"].is_string())  G.voice_region=j["voice_region"];
        if(j.count("voice_variant") && j["voice_variant"].is_string()) G.voice_variant=j["voice_variant"];
        if(j.count("voice_engine")  && j["voice_engine"].is_string())  G.voice_engine=j["voice_engine"];
        if(j.count("voice_stream")  && j["voice_stream"].is_string())  G.voice_stream=j["voice_stream"];
        if(j.count("voice_pitch"))  G.voice_pitch=j["voice_pitch"];
        if(j.count("voice_rate"))   G.voice_rate=j["voice_rate"];
        if(j.count("voice_short"))  G.voice_short=j["voice_short"];
        if(j.count("api_base") && j["api_base"].is_string())
            G.api_base = normalize_api_base(j["api_base"].get<std::string>());
        if(j.count("aliases")) G.aliases=j["aliases"].get<std::unordered_map<std::string,std::string>>();
        if(j.count("session_name") && j["session_name"].is_string())
            G.session_name = j["session_name"].get<std::string>();
    } catch(...){}
}
static void switch_session(const std::string& name) {
    if (G.history_enabled) save_history(true);
    g_undo_stack.clear();   // undo не должен пересекать границу сессии
    G.session_name = name;
    HISTORY_FILE = SESSIONS_DIR + "/" + name + ".json";
    G.history_file = HISTORY_FILE;
    G.messages.clear();
    G.total_prompt_tokens = 0;
    G.total_completion_tokens = 0;
    G.messages.push_back({{"role","system"},{"content",G.sys_prompt}});
    load_history();
    std::cout << C_GREEN << "[Сессия: " << name << "]" << C_RESET << std::endl;
}
static void rename_session(const std::string& new_name) {
    if (new_name.empty()) {
        std::cerr << C_RED << "[Использование: /rename <новое_имя>]" << C_RESET << std::endl;
        return;
    }
    if (new_name.find('/') != std::string::npos ||
        new_name.find('\\') != std::string::npos) {
        std::cerr << C_RED << "[Недопустимое имя сессии]" << C_RESET << std::endl;
        return;
    }
    if (G.history_enabled) save_history(true);
    std::string old_path = SESSIONS_DIR + "/" + G.session_name + ".json";
    std::string new_path = SESSIONS_DIR + "/" + new_name + ".json";
    std::rename(old_path.c_str(), new_path.c_str()); // ок если старого файла нет
    G.session_name = new_name;
    HISTORY_FILE = new_path;
    G.history_file = new_path;
    std::cout << C_GREEN << "[Сессия переименована: " << new_name << "]" << C_RESET << std::endl;
}

static void list_sessions() {
    DIR* dir = opendir(SESSIONS_DIR.c_str());
    if(!dir) { std::cout << C_GRAY << "[Нет сессий]" << C_RESET << std::endl; return; }
    std::cout << C_YELLOW << "[Сессии]:" << C_RESET << std::endl;
    struct dirent* ent; while((ent=readdir(dir))) {
        std::string fn=ent->d_name;
        if(fn.size()>5 && fn.substr(fn.size()-5)==".json") {
            std::string s=fn.substr(0,fn.size()-5);
            std::cout << (s==G.session_name?C_GREEN "► ":"  ") << s << C_RESET << std::endl;
        }
    } closedir(dir);
}
static std::string expand_aliases(const std::string& input) {
    if(input.empty()||(input[0]!='!'&&input[0]!='/')) return input;
    std::string key=input.substr(1); size_t sp=key.find(' ');
    std::string alias=(sp!=std::string::npos)?key.substr(0,sp):key;
    if(G.aliases.count(alias)) {
        std::string val=G.aliases[alias];
        return (sp!=std::string::npos)?val+key.substr(sp):val;
    } return input;
}
static void search_history(const std::string& query) {
    std::string q=query;
    std::transform(q.begin(),q.end(),q.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    bool found=false;
    for(size_t i=0;i<G.messages.size();++i) {
        std::string role = G.messages[i].value("role", std::string(""));
        if (!G.messages[i].count("content") || !G.messages[i]["content"].is_string()) continue;
        std::string cont = G.messages[i]["content"].get<std::string>();
        std::string low=cont;
        std::transform(low.begin(),low.end(),low.begin(),
                       [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
        if(low.find(q)!=std::string::npos) {
            if(cont.size()>150) cont=cont.substr(0,150)+"...";
            std::cout << C_CYAN << "["<<i<<"] "<<role<<": " << C_RESET << cont << std::endl; found=true;
        }
    } if(!found) std::cout << C_GRAY << "[Ничего не найдено]" << C_RESET << std::endl;
}
static std::string strip_ansi(const std::string& s) {
    std::string out; out.reserve(s.size()); size_t i=0;
    while(i<s.size()) {
        if(s[i]=='\033'){++i;if(i<s.size()&&s[i]=='['){++i;while(i<s.size()&&!((s[i]>='@'&&s[i]<='~')))++i;if(i<s.size())++i;}else if(i<s.size())++i;continue;}
        out+=s[i++];
    } return out;
}
static void export_dialog(const std::string& arg) {
    // По умолчанию пишем в ~/tmp/ (а не в CWD). Явно указанный путь — как задан.
    std::string def_dir = get_home_dir() + "/tmp/";
    std::string fmt="md";
    std::string file;
    size_t sp=arg.find(' ');
    if(sp!=std::string::npos){fmt=arg.substr(0,sp);file=arg.substr(sp+1);}
    else if(!arg.empty()){fmt=arg;}
    if(file.empty()) file = def_dir + "dialog_export." + fmt;
    else if(file[0]=='~') file = get_home_dir() + file.substr(1);
    std::ofstream f(file); if(!f.is_open()){std::cerr<<C_RED<<"[Не удалось создать файл: "<<file<<"]"<<C_RESET<<std::endl;return;}
    if (fmt == "json") {
        json arr = json::array();
        for (auto& m : G.messages) arr.push_back(m);
        f << arr.dump(2, ' ', false, json::error_handler_t::replace);
    } else {
        for (auto& m : G.messages) {
            std::string role = m.value("role", std::string("unknown"));
            std::string cont;
            if (m.count("content") && m["content"].is_string())
                cont = m["content"].get<std::string>();
            else if (m.count("content"))
                cont = m["content"].dump();
            std::string txt = (fmt == "txt") ? strip_ansi(cont) : cont;
            f << "## " << role << "\n" << txt << "\n\n";
        }
    }
    std::cout << C_GREEN << "[Экспортировано в " << file << "]" << C_RESET << std::endl;
}

static std::string clip_for_summary(const std::string& s, size_t max_len) {
    if (s.size() <= max_len) return s;
    size_t cut = max_len;
    while (cut > 0 && (s[cut] & 0xC0) == 0x80) --cut;
    return s.substr(0, cut) + "...";
}

static std::string clamp_message_content(const std::string& s, size_t max_len = MAX_MSG_CHARS) {
    if (s.size() <= max_len) return s;
    size_t cut = max_len;
    while (cut > 0 && (s[cut] & 0xC0) == 0x80) --cut;
    std::string out = s.substr(0, cut);
    out += "\n[...сообщение обрезано, лимит " + std::to_string(max_len) + " байт...]";
    if (!is_compact()) {
        std::cout << C_YELLOW << "[Сообщение обрезано: " << s.size()
                  << " -> " << max_len << " байт]" << C_RESET << std::endl;
    }
    return out;
}

static std::string build_local_summary(const std::vector<json>& msgs, int from, int to) {
    std::string out = "[SUMMARY of trimmed context]\n";
    int n = 0;
    for (int i = from; i < to && i < static_cast<int>(msgs.size()); ++i) {
        if (!msgs[i].count("role") || !msgs[i].count("content")) continue;
        if (!msgs[i]["content"].is_string()) continue;
        std::string role = msgs[i]["role"].get<std::string>();
        if (role == "system") continue;
        std::string cont = msgs[i]["content"].get<std::string>();
        std::string flat;
        flat.reserve(cont.size());
        bool sp = false;
        for (size_t k = 0; k < cont.size(); ++k) {
            char c = cont[k];
            if (c == 10 || c == 13 || c == 9 || c == 32) {
                if (!sp) { flat.push_back(32); sp = true; }
            } else {
                flat.push_back(c);
                sp = false;
            }
        }
        out += "- " + role + ": " + clip_for_summary(flat, 180) + "\n";
        if (++n >= 24) {
            out += "- ...\n";
            break;
        }
    }
    if (n == 0) out += "(no user/assistant messages)\n";
    if (out.size() > 3500) out = clip_for_summary(out, 3500);
    return out;
}

static void smart_trim_context() {
    trim_messages_if_needed();   // сначала — ограничение по количеству сообщений
    const size_t MAX_CHARS = (G.max_tokens > 0 ? G.max_tokens : 4096) * 3;
    size_t total = 0;
    for (auto& m : G.messages) {
        if (m.count("content") && m["content"].is_string())
            total += m["content"].get<std::string>().size();
    }
    if (total <= MAX_CHARS) return;

    int sys_idx = -1;
    for (int i = 0; i < static_cast<int>(G.messages.size()); ++i) {
        if (G.messages[i].count("role") && G.messages[i]["role"] == "system") {
            sys_idx = i; break;
        }
    }

    size_t sys_size = 0;
    if (sys_idx >= 0 && G.messages[sys_idx].count("content") &&
        G.messages[sys_idx]["content"].is_string())
        sys_size = G.messages[sys_idx]["content"].get<std::string>().size();

    const size_t SUMMARY_RESERVE = 4000;
    size_t budget = (MAX_CHARS > sys_size + SUMMARY_RESERVE)
        ? (MAX_CHARS - sys_size - SUMMARY_RESERVE)
        : ((MAX_CHARS > sys_size) ? (MAX_CHARS - sys_size) : 0);

    size_t tail_size = 0;
    int split_idx = static_cast<int>(G.messages.size());
    for (int i = static_cast<int>(G.messages.size()) - 1; i >= 0; --i) {
        if (i == sys_idx) continue;
        size_t len = 0;
        if (G.messages[i].count("content") && G.messages[i]["content"].is_string())
            len = G.messages[i]["content"].get<std::string>().size();
        if (tail_size + len > budget) {
            split_idx = i + 1;
            break;
        }
        tail_size += len;
        split_idx = i;
    }

    // Гарантируем, что хотя бы последнее сообщение останется, даже если оно
    // само превышает бюджет (иначе контекст полностью теряется).
    if (split_idx >= static_cast<int>(G.messages.size()) && !G.messages.empty()) {
        split_idx = static_cast<int>(G.messages.size()) - 1;
    }

    int erase_start = (sys_idx >= 0) ? (sys_idx + 1) : 0;
    if (erase_start < static_cast<int>(G.messages.size()) &&
        G.messages[erase_start].count("role") &&
        G.messages[erase_start]["role"] == "system" &&
        G.messages[erase_start].count("content") &&
        G.messages[erase_start]["content"].is_string()) {
        std::string c = G.messages[erase_start]["content"].get<std::string>();
        if (c.rfind("[SUMMARY of trimmed context]", 0) == 0)
            erase_start++;
    }

    if (split_idx > erase_start) {
        std::string summary = build_local_summary(G.messages, erase_start, split_idx);
        if (!is_compact()) {
            std::cout << C_GRAY << "[Trim: " << erase_start << "-" << (split_idx - 1)
                      << " -> summary " << summary.size() << " bytes]" << C_RESET << std::endl;
        }
        G.messages.erase(G.messages.begin() + erase_start, G.messages.begin() + split_idx);
        json sum = {{"role", "system"}, {"content", summary}};
        G.messages.insert(G.messages.begin() + erase_start, sum);
    }
    if (!is_compact()) {
        std::cout << C_GRAY << "[Context optimized: " << G.messages.size()
                  << " msgs, budget ~" << MAX_CHARS << " chars]" << C_RESET << std::endl;
    }
}

static char** cmd_completion(const char* text, int start, int end) {
    (void)end;
    rl_attempted_completion_over = 1;
    std::vector<std::string> matches;
    std::string t(text);
    static const std::vector<std::string> cmds = {"/help","/save","/load","/clear","/history","/delete","/retry","/tokens","/model","/models","/apibase","/temp","/maxtokens","/system","/file","/autorun","/nores","/compact","/cost","/balance","/update","/about","/exit","/new","/list","/switch","/rename","/undo","/alias","/search","/export","/info","/dump","/voice","/speak","/listen","/FIRE","/time"};
    
    try {
        if (start == 0) {
            for (const auto& c : cmds) if (c.rfind(t, 0) == 0) matches.push_back(c);
        } else if (rl_line_buffer && rl_line_buffer[0] == '/' && std::string(rl_line_buffer).rfind("/model ", 0) == 0) {
            // Проверка доступности моделей
            if (!AVAILABLE_MODELS.empty()) {
                for (const auto& m : AVAILABLE_MODELS) {
                    // to_lower_copy объявлен ниже — локальный lower здесь
                    std::string ml = m, tl = t;
                    for (char& c : ml) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                    for (char& c : tl) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                    if (tl.empty() || ml.find(tl) != std::string::npos) matches.push_back(m);
                    if (matches.size() > 40) break;
                }
            }
        }
    } catch (...) {
        return nullptr;
    }

    if (matches.empty()) return nullptr;
    
    char** res = static_cast<char**>(malloc(sizeof(char*) * (matches.size() + 1)));
    if (!res) return nullptr;
    for (size_t i = 0; i < matches.size(); ++i) res[i] = strdup(matches[i].c_str());
    res[matches.size()] = nullptr;
    return res;
}
// D5: readline history — multiline -> one visual line
static std::string history_oneline(const std::string& s, size_t max_len = RL_HIST_MAX_CHARS) {
    std::string h;
    h.reserve(std::min(s.size() + 8, max_len + 8));
    for (size_t i = 0; i < s.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c == '\n') {
            h += " / "; // multiline marker in readline history
        } else if (c == '\r') {
            continue;
        } else if (c == '\t') {
            h += ' ';
        } else {
            h += static_cast<char>(c);
        }
        if (h.size() >= max_len) {
            h += "...";
            break;
        }
    }
    return h;
}

// E5: rough token estimate (chars/4 + per-message overhead)
static size_t approx_tokens_messages() {
    size_t chars = 0;
    size_t msgs = 0;
    for (auto& m : G.messages) {
        ++msgs;
        if (m.count("content") && m["content"].is_string())
            chars += m["content"].get<std::string>().size();
        else if (m.count("content"))
            chars += m["content"].dump().size();
    }
    return chars / 4 + msgs * 6;
}

static size_t approx_context_chars() {
    size_t chars = 0;
    for (auto& m : G.messages) {
        if (m.count("content") && m["content"].is_string())
            chars += m["content"].get<std::string>().size();
    }
    return chars;
}

static std::string to_lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return s;
}

// C3/D4: filter + paginate models list
// a-b/c_d -> "a b c d" чтобы /models claude sonnet находил id с дефисами
static std::string model_id_search_key(const std::string& id) {
    std::string o;
    o.reserve(id.size() + 8);
    for (unsigned char c : id) {
        char ch = static_cast<char>(std::tolower(c));
        if (ch == '-' || ch == '_' || ch == '/' || ch == '.' || ch == ':') o.push_back(' ');
        else o.push_back(ch);
    }
    return o;
}

// Пусто = все. Несколько слов = AND. В токене a|b|c = OR.
static std::vector<std::string> filter_models(const std::string& filt) {
    if (filt.empty()) return AVAILABLE_MODELS;
    std::vector<std::string> tokens;
    {
        std::istringstream ss(filt);
        std::string t;
        while (ss >> t) {
            std::string tl = to_lower_copy(t);
            if (!tl.empty()) tokens.push_back(tl);
        }
    }
    if (tokens.empty()) return AVAILABLE_MODELS;

    std::vector<std::string> out;
    out.reserve(AVAILABLE_MODELS.size() / 4 + 8);
    for (auto& m : AVAILABLE_MODELS) {
        std::string ml = to_lower_copy(m);
        std::string mk = model_id_search_key(m);
        bool ok = true;
        for (auto& tok : tokens) {
            bool any = false;
            if (tok.find('|') != std::string::npos) {
                size_t start = 0;
                while (start <= tok.size()) {
                    size_t bar = tok.find('|', start);
                    std::string alt = (bar == std::string::npos)
                        ? tok.substr(start)
                        : tok.substr(start, bar - start);
                    start = (bar == std::string::npos) ? tok.size() + 1 : bar + 1;
                    if (alt.empty()) continue;
                    if (ml.find(alt) != std::string::npos || mk.find(alt) != std::string::npos) {
                        any = true;
                        break;
                    }
                }
            } else {
                any = (ml.find(tok) != std::string::npos || mk.find(tok) != std::string::npos);
            }
            if (!any) { ok = false; break; }
        }
        if (ok) out.push_back(m);
    }
    return out;
}

// /model arg: exact | N из LAST_MODEL_VIEW | unique filter
// return: 1 ok, 0 fail, 2 ambiguous (LAST_MODEL_VIEW filled)
static int resolve_model_arg(const std::string& arg, std::string& resolved, std::string& err) {
    resolved.clear(); err.clear();
    if (arg.empty()) { err = "пусто"; return -1; }
    for (auto& m : AVAILABLE_MODELS)
        if (m == arg) { resolved = m; return 1; }
    try {
        size_t pos = 0;
        int idx = std::stoi(arg, &pos);
        if (pos == arg.size() && idx >= 1) {
            const auto& view = !LAST_MODEL_VIEW.empty() ? LAST_MODEL_VIEW : AVAILABLE_MODELS;
            if (idx <= static_cast<int>(view.size())) {
                resolved = view[idx - 1];
                return 1;
            }
            err = "номер вне списка (1.." + std::to_string(view.size()) + ")";
            return 0;
        }
    } catch (...) {}
    std::string al = to_lower_copy(arg);
    for (auto& m : AVAILABLE_MODELS)
        if (to_lower_copy(m) == al) { resolved = m; return 1; }
    auto hits = filter_models(arg);
    if (hits.size() == 1) { resolved = hits[0]; return 1; }
    if (hits.empty()) {
        std::vector<std::string> pref;
        for (auto& m : AVAILABLE_MODELS)
            if (to_lower_copy(m).rfind(al, 0) == 0) pref.push_back(m);
        if (pref.size() == 1) { resolved = pref[0]; return 1; }
        if (pref.empty()) { err = "не найдено: " + arg; return 0; }
        LAST_MODEL_VIEW = pref;
        err = "несколько совпадений (" + std::to_string(pref.size()) + "), уточните";
        return 2;
    }
    LAST_MODEL_VIEW = hits;
    err = "несколько совпадений (" + std::to_string(hits.size()) + "), уточните номер или фильтр";
    return 2;
}

// Индикатор заполнения контекста: цветной бар [██████░░░░] NN% + число сообщений.
// Возвращает готовую строку-промпт с \001..\002 (невидимая для readline разметка).
static std::string build_prompt() {
    // compact: minimal prompt, no context bar / hints
    if (is_compact())
        return "\001\033[32m\002\xe2\x9d\xaf \001" C_INPUT "\002";

    size_t chars = 0;
    int msgs = 0;
    for (auto& m : G.messages) {
        if (m.count("content") && m["content"].is_string())
            chars += m["content"].get<std::string>().size();
        ++msgs;
    }
    size_t limit = static_cast<size_t>((G.max_tokens > 0 ? G.max_tokens : 4096)) * 3;
    int pct = static_cast<int>(std::min<size_t>(100, (chars * 100) / (limit > 0 ? limit : 1)));

    // Цвет по заполнению: зелёный < 50 < жёлтый < 80 < красный
    const char* col = (pct < 50) ? "\033[32m" : (pct < 80) ? "\033[33m" : "\033[31m";

    const int W = 10;
    int filled = (pct * W + 50) / 100;          // округление
    std::string bar;
    for (int i = 0; i < W; ++i)
        bar += (i < filled) ? "\xe2\x96\x88"   // █ full block
                            : "\xe2\x96\x91";   // ░ light shade

    // \001..\002 — обёртки невидимых символов для корректного подсчёта длины readline
    char pct_s[8]; snprintf(pct_s, sizeof(pct_s), "%d", pct);
    std::string p;
    p += "\n";
    p += "\001"; p += C_GRAY;   p += "\002"; p += "\xe2\x94\x82 ";          // │
    p += "\001"; p += col;      p += "\002"; p += bar;
    p += "\001"; p += C_GRAY;   p += "\002"; p += " " + std::string(pct_s) + "%";
    p += " \xc2\xb7 " + std::to_string(msgs) + " msg";
    p += " \xc2\xb7 ~" + std::to_string(approx_tokens_messages()) + " tok";
    p += "\001"; p += C_RESET;  p += "\002"; p += "\n";
    p += "\001"; p += C_BOLD;   p += C_GREEN; p += "\002"; p += "\xe2\x9d\xaf "; // ❯
    p += "\001"; p += C_INPUT;  p += "\002";
    return p;
}

// Дата-время по Москве (UTC+3, без перехода на летнее время с 2014).
static std::string moscow_now_str() {
    std::time_t t = std::time(nullptr);
    std::time_t m = t + 3 * 3600;
    std::tm msk{};
    gmtime_r(&m, &msk);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02d.%02d.%04d %02d:%02d:%02d",
                  msk.tm_mday, msk.tm_mon + 1, msk.tm_year + 1900,
                  msk.tm_hour, msk.tm_min, msk.tm_sec);
    return std::string(buf);
}

std::string do_api_request(bool &aborted) {
    aborted = false;
    std::string api_key = get_api_key();
    if (api_key.empty()) return "";
    CURL *curl = curl_easy_init();
    if (!curl) return "";
    smart_trim_context();

    // E5: local estimate after trim
    {
        size_t est = approx_tokens_messages();
        size_t ch = approx_context_chars();
        if (!is_compact() && !G.fire) {
            std::cout << C_GRAY << "[context ~" << est << " tok / " << ch
                      << " chars, max_tokens=" << G.max_tokens << "]" << C_RESET << std::endl;
        }
    }

    // /time on: добавляем дату-время МСК в начало последнего user-сообщения
    // (в КОПИЮ, G.messages и история не меняются; на экран ничего не выводим).
    json messages_out = G.messages;
    if (G.time_prefix && !messages_out.empty()) {
        for (int i = static_cast<int>(messages_out.size()) - 1; i >= 0; --i) {
            if (messages_out[i].value("role", std::string()) == "user") {
                std::string prefix = "[Текущие дата и время (МСК, UTC+3): "
                                     + moscow_now_str() + "]\n";
                messages_out[i]["content"] = prefix +
                    messages_out[i].value("content", std::string());
                break;
            }
        }
    }
    json jData = {
        {"model", G.model},
        {"messages", messages_out},
        {"temperature", G.temperature},
        {"max_tokens", G.max_tokens}
    };
    std::string jsonData = jData.dump(-1, ' ', false, json::error_handler_t::replace);
    std::string chat_url = normalize_api_base(G.api_base) + "/v1/chat/completions";

    struct curl_slist *headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    std::string auth = "Authorization: Bearer " + api_key;
    headers = curl_slist_append(headers, auth.c_str());

    struct StreamState {
        std::string full_content;
        bool header_cleared;
        int retry_after = -1;   // из заголовка Retry-After, если сервер прислал
    };

    auto header_cb = [](char* buffer, size_t size, size_t nitems, void* userp) -> size_t {
        size_t total = size * nitems;
        StreamState* st = static_cast<StreamState*>(userp);
        const std::string key = "retry-after:";
        if (total >= key.size()) {
            std::string lower;
            lower.reserve(key.size());
            for (size_t i = 0; i < key.size(); ++i)
                lower += static_cast<char>(std::tolower(static_cast<unsigned char>(buffer[i])));
            if (lower == key) {
                std::string v(buffer + key.size(), total - key.size());
                while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) v.erase(0, 1);
                while (!v.empty() && (v.back() == '\r' || v.back() == '\n' || v.back() == ' '))
                    v.pop_back();
                try { if (!v.empty()) st->retry_after = std::stoi(v); } catch (...) {}
            }
        }
        return total;
    };

    auto write_cb = [](void *contents, size_t size, size_t nmemb, void *userp) -> size_t {
        if (g_stream_abort) return 0;
        size_t total = size * nmemb;
        StreamState* st = static_cast<StreamState*>(userp);
        if (!st->header_cleared) {
            st->header_cleared = true;
            spinner_stop();
        }
        st->full_content.append(static_cast<char*>(contents), total);
        return total;
    };

    // Auto-fallback: если модель стабильно отдаёт 429/5xx — одна попытка с другой
    static const std::vector<std::string> FALLBACK_MODELS = {
        "deepseek-chat",
        "gpt-5.6-sol",
        "gemini-3.1-pro-preview"
    };

    auto do_attempt = [&](const std::string& model_name, std::string& body_out,
                          long& http_out, CURLcode& curl_out) -> bool {
        StreamState st = {"", false, -1};
        std::string jsonLocal;
        if (model_name == G.model) {
            jsonLocal = jsonData;
        } else {
            json jL = jData;
            jL["model"] = model_name;
            jsonLocal = jL.dump(-1, ' ', false, json::error_handler_t::replace);
        }

        int retries = 3; long backoff = 2; bool ok = false;
        while (retries-- > 0) {
            st.full_content.clear();
            st.retry_after = -1;
            st.header_cleared = false;
            curl_easy_setopt(curl, CURLOPT_URL, chat_url.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonLocal.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(jsonLocal.size()));
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,
                             static_cast<size_t(*)(void*,size_t,size_t,void*)>(write_cb));
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &st);
            curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION,
                             static_cast<size_t(*)(char*,size_t,size_t,void*)>(header_cb));
            curl_easy_setopt(curl, CURLOPT_HEADERDATA, &st);
            curl_easy_setopt(curl, CURLOPT_TIMEOUT, 420L);
            curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 15L);
            curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
            // Включаем progress-колбэк — он позволяет прервать запрос
            // по Ctrl+C ещё до прихода первых байт ответа.
            curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
            curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, curl_progress_cb);
            spinner_start(model_name);
            CURLcode r = curl_easy_perform(curl);
            spinner_stop();
            long hc = 0;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &hc);
            if (hc == 200 && r == CURLE_OK) {
                http_out = hc; curl_out = r; body_out = st.full_content;
                ok = true; break;
            }
            if (g_stream_abort) {
                http_out = hc; curl_out = r; body_out = st.full_content;
                break;
            }
            bool retryable = (r == CURLE_OPERATION_TIMEDOUT || r == CURLE_COULDNT_CONNECT ||
                              (hc >= 500 && hc < 600) || hc == 429 || hc == 408);
            if (!retryable || retries == 0) {
                http_out = hc; curl_out = r; body_out = st.full_content;
                break;
            }
            long wait_s = backoff;
            if (st.retry_after > 0 && st.retry_after <= 60) wait_s = st.retry_after;
            if (!is_compact())
                std::cout << "\r\033[2K" << C_YELLOW << "[Ошибка сети, повтор через "
                          << wait_s << "с... (Ctrl+C — отмена)]" << C_RESET << std::flush;
            // Дробим паузу на слайсы по 200 мс — Ctrl+C прерывает ожидание сразу.
            {
                bool aborted_wait = false;
                for (long ms = 0; ms < wait_s * 1000; ms += 200) {
                    if (g_stream_abort) { aborted_wait = true; break; }
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                }
                if (aborted_wait || g_stream_abort) break;
            }
            backoff *= 2;
        }
        return ok;
    };

    { std::lock_guard<std::mutex> lock(g_stream_mutex); g_stream_abort = 0; g_in_streaming = 1; }

    StreamState state = {"", false, -1};
    long http_code = 0; CURLcode res = CURLE_OK;
    bool ok = do_attempt(G.model, state.full_content, http_code, res);

    std::string used_model = G.model;
    if (!ok && !g_stream_abort &&
        (http_code == 429 || (http_code >= 500 && http_code < 600))) {
        if (!is_compact())
            std::cout << C_YELLOW << "[Auto-fallback: " << G.model
                      << " недоступна (HTTP " << http_code
                      << "), пробую другую модель...]" << C_RESET << std::endl;
        for (auto& fm : FALLBACK_MODELS) {
            if (fm == G.model || fm == used_model) continue;
            std::string body; long hc = 0; CURLcode cr = CURLE_OK;
            if (do_attempt(fm, body, hc, cr) && hc == 200) {
                state.full_content = body;
                http_code = hc; res = cr;
                used_model = fm;
                if (!is_compact())
                    std::cout << C_GREEN << "[Auto-fallback OK: " << fm << "]"
                              << C_RESET << std::endl;
                break;
            }
            if (g_stream_abort) break;
        }
    }

    spinner_stop();
    { std::lock_guard<std::mutex> lock(g_stream_mutex); g_in_streaming = 0; }

    bool was_aborted = false;
    { std::lock_guard<std::mutex> lock(g_stream_mutex); if (g_stream_abort) { was_aborted = true; g_stream_abort = 0; } }
    if (was_aborted) { aborted = true; if (!is_compact()) std::cout << "\n" << C_YELLOW << "[Запрос прерван]" << C_RESET << std::endl; curl_slist_free_all(headers); curl_easy_cleanup(curl); return ""; }
    if (res != CURLE_OK) { std::cerr << C_RED << "curl: " << curl_easy_strerror(res) << C_RESET << std::endl; curl_slist_free_all(headers); curl_easy_cleanup(curl); return ""; }
    if (http_code != 200) { std::cerr << C_RED << "[HTTP " << http_code << "] " << state.full_content.substr(0, 300) << C_RESET << std::endl; curl_slist_free_all(headers); curl_easy_cleanup(curl); return ""; }

    // Информируем пользователя, если реально отвечала fallback-модель
    if (used_model != G.model && !is_compact()) {
        std::cout << C_GRAY << "[ответ получен от: " << used_model << "]"
                  << C_RESET << std::endl;
    }

    {
        try {
            json j = json::parse(state.full_content);
            if (j.count("choices") && !j["choices"].empty()) {
                auto& choice = j["choices"][0];
                if (choice.count("message") && choice["message"].count("content"))
                    state.full_content = choice["message"]["content"].get<std::string>();
            }
            if (j.count("usage")) {
                G.total_prompt_tokens += j["usage"].value("prompt_tokens", 0);
                G.total_completion_tokens += j["usage"].value("completion_tokens", 0);
            }
        } catch (...) {}
    }
    curl_slist_free_all(headers); curl_easy_cleanup(curl);
    return sanitize_utf8(state.full_content);
}

// ─────────────────────────── Обработка ответа ────────────────
// Выводит ответ красиво, обрабатывает bash-команды
// aborted — если ответ был прерван, не добавляем его в историю
void process_response(const std::string &content, bool aborted, size_t msgs_before = 0) {
    if (content.empty()) return;

    if (aborted) {
        print_assistant_text(content);
        const char* prompt = is_compact()
            ? C_YELLOW "(y/n) " C_RESET
            : C_YELLOW "[Ответ прерван. Сохранить в историю? (y/n | д/н)]: " C_RESET;
        char *rl_ans = readline(prompt);
        std::string ans;
        if (rl_ans) { ans = std::string(rl_ans); free(rl_ans); }
        if (ans != "y" && ans != "Y" && ans != "д" && ans != "Д") {
            if (msgs_before > 0 && msgs_before <= G.messages.size()) {
                G.messages.resize(msgs_before);
            } else if (!G.messages.empty() && G.messages.back().value("role", "") == "user") {
                G.messages.pop_back();
            }
            note_gray("[Частичный ответ отброшен]");
            return;
        }
        G.messages.push_back({{"role", "assistant"}, {"content", content}});
        return;
    }

    // ── Ищем bash-блоки в ответе (```bash / ```sh, пробелы в info-string) ──
    auto find_closing = [](const std::string &text, size_t from) -> size_t {
        size_t pos = from;
        while (pos < text.size()) {
            auto p = text.find("```", pos);
            if (p == std::string::npos) return std::string::npos;
            auto after = p + 3;
            if (after >= text.size() || text[after] == '\n' ||
                text[after] == '\r'  || text[after] == ' ') {
                return p;
            }
            pos = p + 3;
        }
        return std::string::npos;
    };

    // true if fence language is bash/sh (case-insensitive), allows spaces
    auto is_exec_fence = [](const std::string &text, size_t ts, size_t &code_start) -> bool {
        if (ts + 3 > text.size() || text.compare(ts, 3, "```") != 0) return false;
        size_t i = ts + 3;
        while (i < text.size() && (text[i] == ' ' || text[i] == '\t')) ++i;
        auto is_alpha = [](char c) {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
        };
        size_t lang_s = i;
        while (i < text.size() && is_alpha(text[i])) ++i;
        if (lang_s == i) return false;
        std::string lang = text.substr(lang_s, i - lang_s);
        for (char &c : lang) if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
        if (lang != "bash" && lang != "sh") return false;
        while (i < text.size() && text[i] != '\n' && text[i] != '\r') ++i;
        if (i < text.size() && text[i] == '\r') ++i;
        if (i < text.size() && text[i] == '\n') ++i;
        code_start = i;
        return true;
    };

    struct BBlock { size_t tag_s, code_s, code_e, blk_e; std::string code; };
    auto find_bash_blocks = [&](const std::string &text) {
        std::vector<BBlock> bbs;
        size_t pos = 0;
        while (pos < text.size()) {
            auto ts = text.find("```", pos);
            if (ts == std::string::npos) break;
            size_t cs = 0;
            if (!is_exec_fence(text, ts, cs)) { pos = ts + 3; continue; }
            auto ce = find_closing(text, cs);
            if (ce == std::string::npos) break;
            auto be = ce + 3;
            if (be < text.size() && text[be] == '\n') be++;
            std::string code = text.substr(cs, ce - cs);
            bbs.push_back({ts, cs, ce, be, code});
            pos = be;
        }
        return bbs;
    };

    // chain_autorun живёт на уровне всего process_response и сохраняет состояние
    // между вызовами render_and_execute, чтобы 'a' покрывало всю цепочку
    // (все ответы модели до MAX_BASH_CHAIN), а не только текущий ответ.
    // Сбрасывается автоматически при следующем вводе пользователя — process_response
    // вызывается заново, и переменная создаётся со значением false.
    bool chain_autorun = false;

    // ── Функция: вывести ответ по частям, останавливаясь на bash-блоках ──
    // leftover — текст после последнего bash-блока (не рендерится, ждёт результатов)
    auto render_and_execute = [&](const std::string &text, std::string &leftover) -> std::string {
        leftover = "";
        // Trim trailing whitespace/newlines for rendering and bash detection
        std::string t = text;
        while (!t.empty() && (t.back() == '\n' || t.back() == '\r' || t.back() == ' ')) t.pop_back();
        
        auto bbs = find_bash_blocks(t);
        if (bbs.empty()) {
            print_assistant_text(t);
            return "";
        }

        if (!is_compact() && !G.fire)
            std::cout << "\n" << C_BOLD << C_CYAN << "[Ассистент]:" << C_RESET << "\n";

        std::string combined_result;
        size_t cur = 0;
        int total = static_cast<int>(bbs.size());
        // chain_autorun захвачен снаружи по ссылке (см. объявление выше).

        for (int i = 0; i < total; ++i) {
            // Текст до bash-блока
            if (bbs[i].tag_s > cur) {
                std::string chunk = text.substr(cur, bbs[i].tag_s - cur);
                if (G.fire) {
                    while (!chunk.empty() &&
                           (chunk.back() == '\n' || chunk.back() == '\r' || chunk.back() == ' '))
                        chunk.pop_back();
                }
                if (!chunk.empty()) {
                    render_markdown(chunk);
                }
            }
            // Сам bash-блок (визуально) — в FIRE полностью скрыт.
            if (!G.fire) {
                std::string chunk = text.substr(bbs[i].tag_s, bbs[i].blk_e - bbs[i].tag_s);
                if (!chunk.empty()) render_markdown(chunk);
            }
            std::cout << std::flush;

            // Выполняем (флаг chain_autorun общий для всей цепочки)
            std::string res = execute_single_bash(bbs[i].code, i, total, chain_autorun);
            if (!res.empty()) {
                if (!combined_result.empty()) combined_result += "\n---\n";
                if (total > 1) combined_result += "[Блок " + std::to_string(i+1) + "]:\n";
                combined_result += res;
            }
            cur = bbs[i].blk_e;
        }

        // Текст после последнего блока — НЕ рендерим, сохраняем как leftover
        if (cur < t.size()) {
            leftover = t.substr(cur);
        }
        if (!is_compact() && !G.fire) std::cout << std::endl;
        return combined_result;
    };

    // ── Основная логика ──
    // FIX(BUG3): раньше leftover был один слот и перетирался на каждой
    // итерации цепочки — текст ПОСЛЕ bash-блока у всех ответов кроме
    // последнего молча терялся. Теперь аккумулируем в вектор и печатаем
    // все в исходном порядке после завершения цепочки.
    std::vector<std::string> leftovers;
    std::string leftover;
    std::string cmd_result = render_and_execute(content, leftover);
    if (!leftover.empty()) leftovers.push_back(leftover);

    // В контекст сохраняем весь ответ целиком (leftover будет показан после результатов)
    G.messages.push_back({{"role", "assistant"}, {"content", content}});

    // Цикл: если были bash-результаты, отправляем модели
    bool chain_error = false, chain_aborted_flag = false;
    for (int chain = 0; chain < MAX_BASH_CHAIN && !cmd_result.empty(); ++chain) {
        // D8: chain progress
        if (!is_compact() && !G.fire) {
            std::cout << C_CYAN << C_BOLD << "[bash chain " << (chain + 1)
                      << "/" << MAX_BASH_CHAIN << "]" << C_RESET << std::endl;
        }
        // Отправляем результат bash модели
        std::string user_msg = "[Результат выполнения команды]:\n" + cmd_result;
        G.messages.push_back({{"role", "user"}, {"content", user_msg}});

        bool chain_aborted = false;
        std::string next = do_api_request(chain_aborted);
        if (next.empty()) {
            // API не ответил — оставляем результат bash в контексте,
            // чтобы модель увидела его при следующем запросе (не теряем вывод).
            std::cerr << C_RED
                      << "[Ответ модели не получен. Результат bash сохранён в контексте."
                      << " Повторить: /retry]"
                      << C_RESET << std::endl;
            chain_error = true;
            break;
        }

        if (chain_aborted) {
            print_assistant_text(next);
            note_yellow("[Ответ прерван]");
            chain_aborted_flag = true;
            break;
        }

        std::string next_leftover;
        cmd_result = render_and_execute(next, next_leftover);
        G.messages.push_back({{"role", "assistant"}, {"content", next}});
        // FIX(BUG3): накапливаем хвост каждого ответа, не перетираем.
        if (!next_leftover.empty()) leftovers.push_back(next_leftover);
    }

    // FIX(BUG3): печатаем хвосты ВСЕХ ответов цепочки (а не только последнего).
    if (!chain_aborted_flag) {
        for (auto lo : leftovers) {
            if (G.fire) {
                size_t k = 0;
                while (k < lo.size() &&
                       (lo[k] == '\n' || lo[k] == '\r' || lo[k] == ' ')) ++k;
                lo.erase(0, k);
            }
            if (!lo.empty()) print_assistant_text(lo, false);
        }
    }

    // Явный маркер, если цепочка не смогла продолжиться.
    if (!cmd_result.empty() && !chain_aborted_flag) {
        if (chain_error)
            note_yellow("[bash chain прерван ошибкой API — остаток не отправлен]");
        else
            note_yellow("[bash chain: достигнут лимит " + std::to_string(MAX_BASH_CHAIN)
                        + " шагов, оставшийся вывод не отправлен]");
    }

    // Автосохранение: пишем всегда, уведомляем раз в 24 сообщения
    if (G.history_enabled && G.messages.size() > 2) save_history(G.messages.size() % 24 != 0);
}

// ─────────────────────────── Команды ──────────────────────────
void cmd_update() {
    if (g_dry_run) {
        std::cout << C_YELLOW << "[DRY-RUN: /update пропущен (обновление не выполнено)]"
                  << C_RESET << std::endl;
        return;
    }
    std::string home = get_home_dir();
    std::string url = "https://raw.githubusercontent.com/swarik/Chat-Assist/main/sw_chat.cpp";
    std::string new_src = home + "/tmp/sw_chat_new.cpp";
    std::string new_bin = home + "/tmp/sw_chat_new";
    // Единый источник истины: где реально установлен бинарник (через /proc/self/exe).
    auto find_current_bin = [&home]() -> std::string {
        char buf[4096];
        ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
        if (n > 0) {
            buf[n] = '\0';
            std::string p(buf);
            if (access(p.c_str(), X_OK) == 0) return p;
        }
        std::vector<std::string> cands = {
            home + "/sw_chat",
            home + "/.local/bin/sw_chat"
        };
        for (auto& p : cands) if (access(p.c_str(), X_OK) == 0) return p;
        return home + "/sw_chat";
    };
    std::string cur_bin = find_current_bin();
    std::cout << C_GRAY << "[update] Текущий бинарник: " << cur_bin << C_RESET << std::endl;

    // 1. Скачать новый исходник
    std::cout << C_YELLOW << "[update] Скачиваю обновление..." << C_RESET << std::endl;

    CURL *curl = curl_easy_init();
    if (!curl) {
        std::cerr << C_RED << "[update: curl init failed]" << C_RESET << std::endl;
        return;
    }

    std::string src_body;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &src_body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        std::cerr << C_RED << "[update: download failed: " << curl_easy_strerror(res) << "]" << C_RESET << std::endl;
        return;
    }
    if (http_code != 200) {
        std::cerr << C_RED << "[update: HTTP " << http_code << "]" << C_RESET << std::endl;
        return;
    }
    if (src_body.size() < 100) {
        std::cerr << C_RED << "[update: файл слишком маленький, прерывание]" << C_RESET << std::endl;
        return;
    }

    // 2. Проверить версию — извлечь APP_VERSION из скачанного файла
    auto extract_version = [](const std::string& src) -> std::string {
        std::string marker = "#define APP_VERSION \"";
        size_t pos = src.find(marker);
        if (pos == std::string::npos) return "";
        pos += marker.size();
        size_t end = src.find("\"", pos);
        if (end == std::string::npos) return "";
        return src.substr(pos, end - pos);
    };

    // Сравнение версий "1.2.3" > "1.0.3" и т.д.
    auto version_greater = [](const std::string& remote, const std::string& local) -> bool {
        auto split = [](const std::string& s) {
            std::vector<int> parts;
            std::stringstream ss(s);
            std::string token;
            while (std::getline(ss, token, '.')) {
                // take leading digits only: "37-beta" -> 37
                size_t i = 0;
                while (i < token.size() && token[i] >= '0' && token[i] <= '9') ++i;
                if (i == 0) { parts.push_back(0); continue; }
                try { parts.push_back(std::stoi(token.substr(0, i))); }
                catch (...) { parts.push_back(0); }
            }
            return parts;
        };
        auto rv = split(remote);
        auto lv = split(local);
        size_t n = std::max(rv.size(), lv.size());
        while (rv.size() < n) rv.push_back(0);
        while (lv.size() < n) lv.push_back(0);
        for (size_t i = 0; i < n; i++) {
            if (rv[i] > lv[i]) return true;
            if (rv[i] < lv[i]) return false;
        }
        return false;
    };

    std::string remote_ver = extract_version(src_body);
    std::string local_ver  = APP_VERSION;

    std::cout << C_GRAY << "[update] Локальная версия:  " << local_ver << C_RESET << std::endl;
    std::cout << C_GRAY << "[update] Удалённая версия: " << remote_ver << C_RESET << std::endl;

    if (remote_ver.empty()) {
        std::cerr << C_RED << "[update: не удалось определить версию на сервере]" << C_RESET << std::endl;
        return;
    }

    if (!version_greater(remote_ver, local_ver)) {
        std::cout << C_GREEN << "[update] Уже последняя версия (" << local_ver << ")!" << C_RESET << std::endl;
        return;
    }

    std::cout << C_YELLOW << "[update] Доступна новая версия: " << remote_ver << C_RESET << std::endl;

    // D6: brief colored "diff" / changelog preview vs local source
    {
        std::string local_src;
        {
            // try common locations
            std::vector<std::string> cands = {
                home + "/tmp/sw_chat.cpp",
                home + "/sw_chat.cpp"
            };
            for (auto& pth : cands) {
                std::ifstream lf(pth);
                if (!lf.is_open()) continue;
                local_src.assign((std::istreambuf_iterator<char>(lf)), std::istreambuf_iterator<char>());
                if (!local_src.empty()) break;
            }
        }
        auto interesting = [](const std::string& line) -> bool {
            if (line.find("APP_VERSION") != std::string::npos) return true;
            if (line.find("void cmd_") != std::string::npos) return true;
            if (line.find("static void cmd_") != std::string::npos) return true;
            if (line.find("/help") != std::string::npos) return true;
            if (line.find("void print_help") != std::string::npos) return true;
            if (line.find("#define ") != std::string::npos) return true;
            if (line.find("CHANGELOG") != std::string::npos || line.find("Changelog") != std::string::npos) return true;
            if (line.find("api_base") != std::string::npos) return true;
            return false;
        };
        std::unordered_set<std::string> local_lines;
        {
            std::istringstream ss(local_src);
            std::string line;
            while (std::getline(ss, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (interesting(line)) local_lines.insert(line);
            }
        }
        std::vector<std::string> added;
        {
            std::istringstream ss(src_body);
            std::string line;
            while (std::getline(ss, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (!interesting(line)) continue;
                if (!local_lines.count(line)) added.push_back(line);
            }
        }
        std::cout << C_GRAY << "[update] remote size: " << src_body.size()
                  << " bytes (local src: " << local_src.size() << ")" << C_RESET << std::endl;
        if (added.empty()) {
            std::cout << C_GRAY << "[update] changelog: no marked feature-lines delta (version bump only?)"
                      << C_RESET << std::endl;
        } else {
            std::cout << C_GREEN << "[update] new/interesting lines (+"
                      << added.size() << ", show up to 20):" << C_RESET << std::endl;
            size_t shown = 0;
            for (auto& ln : added) {
                std::string s = ln;
                if (s.size() > 120) s = s.substr(0, 117) + "...";
                std::cout << C_GREEN << "  + " << C_RESET << s << std::endl;
                if (++shown >= 20) break;
            }
        }
    }

    // Запрашиваем согласие пользователя
    {
        const char* upd_prompt = is_compact()
            ? C_YELLOW "(y/n) " C_RESET
            : C_YELLOW "[update] Установить обновление? (y/n | д/н): " C_RESET;
        char *rl_ans = readline(upd_prompt);
        std::string ans;
        if (rl_ans) { ans = std::string(rl_ans); free(rl_ans); }
        if (ans != "y" && ans != "Y" && ans != "д" && ans != "Д") {
            std::cout << C_GRAY << "[update] Обновление отменено пользователем]" << C_RESET << std::endl;
            return;
        }
    }

    // 3. Сохранить новый исходник
    {
        std::ofstream out(new_src);
        if (!out.is_open()) {
            std::cerr << C_RED << "[update: не удалось сохранить " << new_src << "]" << C_RESET << std::endl;
            return;
        }
        out << src_body;
        out.close();
    }

    // 4. Скомпилировать
    std::cout << C_YELLOW << "[update] Компиляция..." << C_RESET << std::endl;
    std::string compile_cmd =
        "CXX=\"${CXX:-g++}\"; command -v \"$CXX\" >/dev/null 2>&1 || CXX=clang++; "
        "\"$CXX\" -std=c++17 -O2 -I" + shell_escape(home + "/.local/include") + " -o "
        + shell_escape(new_bin) + " " + shell_escape(new_src) + " -lreadline -lcurl -lpthread 2>&1";
    std::string compile_out;
    {
        FILE *pipe = popen(compile_cmd.c_str(), "r");
        if (pipe) {
            char buf[256];
            while (fgets(buf, sizeof(buf), pipe)) compile_out += buf;
            int status = pclose(pipe);
            if (status != 0) {
                std::cerr << C_RED << "[update: компиляция не удалась]" << C_RESET << std::endl;
                std::cerr << compile_out << std::endl;
                return;
            }
        }
    }

    // 5. Проверить что бинарник создан
    if (access(new_bin.c_str(), X_OK) != 0) {
        std::cerr << C_RED << "[update: бинарник не создан]" << C_RESET << std::endl;
        return;
    }

    // 6. Сохранить историю перед рестартом
    std::cout << C_YELLOW << "[update] Сохраняю историю..." << C_RESET << std::endl;
    save_history();

    // P0.2: перед подменой убеждаемся, что новый бинарь ЗАПУСКАЕТСЯ и отвечает на --version.
    // Если новый бинарь нерабочий — отказ без подмены рабочего.
    {
        std::string probe_cmd = shell_escape(new_bin) + " --version >/dev/null 2>&1";
        int probe_res = system(probe_cmd.c_str());
        if (probe_res != 0) {
            std::cerr << C_RED << "[update: новый бинарь не запускается, обновление отменено]"
                      << C_RESET << std::endl;
            return;
        }
    }

    // 7. Замена. Порядок безопасный:
    //    - старый бинарь сохраняем как .old (не удаляем — оставляем возможность отката);
    //    - бинарь подменяем атомарным mv;
    //    - исходник копируем (не mv) — исходник остаётся для диффов/бэкапа.
    std::string old_bin = cur_bin + ".old";
    std::string src_path = home + "/sw_chat.cpp";

    auto run = [](const std::string& cmd) -> int { return system(cmd.c_str()); };

    // 7a. Снести предыдущий .old (если был) и сохранить текущий бинарь как .old.
    std::string backup_cmd = "rm -f " + shell_escape(old_bin) + " && cp -a "
        + shell_escape(cur_bin) + " " + shell_escape(old_bin);
    if (run(backup_cmd) != 0) {
        std::cerr << C_RED << "[update: не удалось создать резервную копию " << old_bin
                  << "]" << C_RESET << std::endl;
        return;
    }

    // 7b. Заменить бинарь атомарно.
    std::string replace_bin_cmd = "mv " + shell_escape(new_bin) + " " + shell_escape(cur_bin)
        + " && chmod +x " + shell_escape(cur_bin);
    if (run(replace_bin_cmd) != 0) {
        std::cerr << C_RED << "[update: не удалось заменить бинарь]" << C_RESET << std::endl;
        return;
    }

    // 7c. Скопировать исходник (сначала во временный файл рядом, затем атомарный mv).
    std::string src_tmp = src_path + ".new";
    {
        std::string cp_cmd = "cp -a " + shell_escape(new_src) + " " + shell_escape(src_tmp)
            + " && mv " + shell_escape(src_tmp) + " " + shell_escape(src_path);
        if (run(cp_cmd) != 0) {
            std::cerr << C_YELLOW
                      << "[update: бинарь обновлён, но исходник не скопирован в "
                      << src_path << "]"
                      << C_RESET << std::endl;
        }
    }

    std::cout << C_GRAY << "[update: бэкап бинаря сохранён: " << old_bin
              << " (можно удалить вручную)]" << C_RESET << std::endl;

    std::cout << C_GREEN << C_BOLD << "[update] Обновление установлено! Перезапуск..." << C_RESET << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // 9. exec — заменить текущий процесс
    execl(cur_bin.c_str(), cur_bin.c_str(), "--restore-session", static_cast<char*>(NULL));

    // Если execl не сработал
    std::cerr << C_RED << "[update: exec failed]" << C_RESET << std::endl;
}

void cmd_balance() {
    // 302.ai: no OpenRouter-style credits API in this client.
    // Honest status + local session token stats (never print full key).
    std::string api_key = get_api_key();
    std::cout << C_CYAN << C_BOLD << "  === Balance / Usage ===" << C_RESET << std::endl;
    if (api_key.empty()) {
        std::cout << C_RED << "  API key: not found" << C_RESET << std::endl;
    } else {
        std::string tail = api_key.size() > 4 ? api_key.substr(api_key.size() - 4) : api_key;
        std::cout << "  API key: " << C_GREEN << "ok" << C_RESET
                  << C_GRAY << " (....." << tail << ")" << C_RESET << std::endl;
    }
    std::cout << "  Provider: " << C_GRAY << normalize_api_base(G.api_base) << C_RESET << std::endl;
    std::cout << "  Balance:  " << C_YELLOW << "check in 302.ai dashboard" << C_RESET << std::endl;
    std::cout << "  Session:  " << C_GREEN << G.total_prompt_tokens << C_RESET << " prompt + "
              << C_GREEN << G.total_completion_tokens << C_RESET << " completion" << std::endl;
}

void cmd_about() {
    std::cout << C_CYAN << C_BOLD << "  === Chat CLI ===" << C_RESET << std::endl;
    std::cout << "  Версия:   " << C_GREEN << APP_VERSION << C_RESET << std::endl;
    std::cout << "  Модель:   " << C_GREEN << G.model << C_RESET << std::endl;
    std::cout << "  Temp:     " << C_GREEN << G.temperature << C_RESET << std::endl;
    std::cout << "  Max tok:  " << C_GREEN << G.max_tokens << C_RESET << std::endl;
    std::cout << "  Autorun:  " << (G.autorun ? C_GREEN "вкл" : C_RED "выкл") << C_RESET << std::endl;
    std::cout << "  History:  " << (G.history_enabled ? C_GREEN "вкл" : C_RED "выкл") << C_RESET << std::endl;
    std::cout << "  NoRes:    " << (G.nores ? C_RED "вкл" : C_GREEN "выкл") << " (скрытие вывода bash)" << C_RESET << std::endl;
    std::cout << "  Compact:  " << (G.compact_mode ? C_GREEN "вкл" : C_RED "выкл") << C_RESET << std::endl;
    std::cout << "  FIRE:     " << (G.fire ? C_RED "ВКЛ 🔥" : C_GRAY "выкл") << C_RESET << std::endl;
    std::cout << "  Time:     " << (G.time_prefix ? C_GREEN "вкл" : C_GRAY "выкл") << C_RESET << " (дата-время МСК в начале запроса)" << std::endl;
    std::cout << "  API base: " << C_GREEN << normalize_api_base(G.api_base) << C_RESET << std::endl;

    std::cout << "  Msgs:     " << C_GREEN << G.messages.size() << C_RESET << std::endl;
    std::cout << "  Tokens:   " << C_GREEN << G.total_prompt_tokens << C_RESET << " prompt + "
              << C_GREEN << G.total_completion_tokens << C_RESET << " completion" << std::endl;
    if (G.history_enabled) {
        std::cout << "  Session:  " << C_GREEN << G.session_name << C_RESET << std::endl;
    }
    std::cout << C_YELLOW << "  Paths:" << C_RESET << std::endl;
    std::cout << "  Config:   " << C_GRAY << CONFIG_FILE << C_RESET << std::endl;
    std::cout << "  Sessions: " << C_GRAY << SESSIONS_DIR << C_RESET << std::endl;
    std::cout << "  History:  " << C_GRAY << HISTORY_FILE << C_RESET << std::endl;
    std::cout << "  Readline: " << C_GRAY << READLINE_HIST_FILE << C_RESET << std::endl;
    cmd_balance();
}

// ─────────────────────────── Сигнал / выход ──────────────────
// ── Подтверждение выхода ──
// Возвращает true, если выходить можно (подтверждено или спрашивать не нужно).
// В неинтерактивном режиме (pipe/--exec) вопрос не задаётся — скрипт не должен висеть.
static bool g_exit_confirmed = false;

static bool confirm_exit() {
    if (g_exit_confirmed) return true;
    if (!isatty(fileno(stdin))) return true;   // неинтерактивно — выходим молча

    const char* prompt = is_compact()
        ? C_YELLOW "(y/n) " C_RESET
        : C_YELLOW "[Выходите из программы? (y/n | д/н)]: " C_RESET;

    std::cout.flush(); fflush(stdout);
    char *rl = readline(prompt);
    if (!rl) {
        // EOF (Ctrl+D) или сигнал — спросить нельзя, выходим без зацикливания.
        g_exit_confirmed = true;
        std::cout << std::endl;
        return true;
    }
    std::string ans(rl); free(rl);

    // Пустой ввод = "нет" (безопаснее случайного выхода).
    bool yes = (ans == "y" || ans == "Y" || ans == "yes" || ans == "Yes" ||
                ans == "д" || ans == "Д" || ans == "да" || ans == "Да");
    if (yes) { g_exit_confirmed = true; return true; }
    return false;
}

void do_exit() {
    // Спрашиваем подтверждение; если отказались — отменяем выход.
    if (!confirm_exit()) {
        g_exit_requested = 0;
        g_stream_abort   = 0;
        g_in_streaming   = 0;
        if (!is_compact())
            std::cout << C_GRAY << "[Выход отменён]" << C_RESET << std::endl;
        return;
    }

    g_exit_requested = 1;
    spinner_stop();

    save_config();
    if (G.history_enabled) {
        std::cout << "\n" << C_YELLOW << "[Сохраняю историю...]" << C_RESET << std::endl;
        save_history();
        write_history(READLINE_HIST_FILE.c_str());
    }
    curl_global_cleanup();
    std::cout << C_YELLOW << "[Выход.]" << C_RESET << std::endl;
    exit(0);
}

// ─────────────────────────── Справка ─────────────────────────
// ─────────────────────── Список моделей ──────────────────────


// ─────────────────────────── Models cache / live /v1/models ───────────
static double json_price_to_per_mtok(const json& v); // defined near parse_models_json
static void save_models_cache(const std::vector<std::string>& models) {
    try {
        json j;
        j["updated"] = static_cast<int>(time(nullptr));
        j["models"] = models;
        if (!MODEL_PRICING_LIVE.empty()) {
            json pr = json::object();
            for (auto& kv : MODEL_PRICING_LIVE) {
                pr[kv.first] = { {"prompt", kv.second.first}, {"completion", kv.second.second} };
            }
            j["pricing"] = pr;
        }
        std::ofstream f(MODELS_CACHE_FILE);
        if (f.is_open()) f << j.dump(2);
    } catch (...) {}
}

static bool load_models_cache() {
    std::ifstream f(MODELS_CACHE_FILE);
    if (!f.is_open()) return false;
    try {
        std::string c((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (c.empty()) return false;
        json j = json::parse(c);
        if (!j.count("models") || !j["models"].is_array()) return false;
        std::vector<std::string> models;
        for (auto& m : j["models"]) {
            if (m.is_string()) {
                std::string id = m.get<std::string>();
                if (!id.empty()) models.push_back(id);
            } else if (m.is_object() && m.count("id") && m["id"].is_string()) {
                models.push_back(m["id"].get<std::string>());
            }
        }
        if (models.empty()) return false;
        AVAILABLE_MODELS = models;
        if (j.count("pricing") && j["pricing"].is_object()) {
            for (auto it = j["pricing"].begin(); it != j["pricing"].end(); ++it) {
                try {
                    double pp = -1, cc = -1;
                    if (it.value().is_object()) {
                        if (it.value().count("prompt")) pp = json_price_to_per_mtok(it.value()["prompt"]);
                        if (it.value().count("completion")) cc = json_price_to_per_mtok(it.value()["completion"]);
                    }
                    if (pp >= 0 && cc >= 0) MODEL_PRICING_LIVE[it.key()] = {pp, cc};
                } catch (...) {}
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

static double json_price_to_per_mtok(const json& v) {
    // Accept: 0.26 meaning $/MTok, or micro-prices; prefer explicit object fields.
    try {
        if (v.is_number()) return v.get<double>();
        if (v.is_string()) return std::stod(v.get<std::string>());
    } catch (...) {}
    return -1.0;
}

static void extract_pricing_from_model_obj(const json& m, const std::string& id) {
    if (id.empty() || !m.is_object()) return;
    double p = -1, c = -1;
    if (m.count("pricing") && m["pricing"].is_object()) {
        const auto& pr = m["pricing"];
        if (pr.count("prompt")) p = json_price_to_per_mtok(pr["prompt"]);
        if (pr.count("completion")) c = json_price_to_per_mtok(pr["completion"]);
        if (pr.count("input")) p = json_price_to_per_mtok(pr["input"]);
        if (pr.count("output")) c = json_price_to_per_mtok(pr["output"]);
    }
    if (p < 0 && m.count("prompt_price")) p = json_price_to_per_mtok(m["prompt_price"]);
    if (c < 0 && m.count("completion_price")) c = json_price_to_per_mtok(m["completion_price"]);
    // Some APIs give price per token; if values look tiny, scale to 1M
    if (p > 0 && p < 0.0001) p *= 1000000.0;
    if (c > 0 && c < 0.0001) c *= 1000000.0;
    if (p >= 0 && c >= 0)
        MODEL_PRICING_LIVE[id] = {p, c};
}

static std::vector<std::string> parse_models_json(const std::string& body) {
    std::vector<std::string> models;
    json j = json::parse(body);
    json arr = json::array();
    if (j.is_array()) arr = j;
    else if (j.count("data") && j["data"].is_array()) arr = j["data"];
    else if (j.count("models") && j["models"].is_array()) arr = j["models"];
    for (auto& m : arr) {
        std::string id;
        if (m.is_string()) id = m.get<std::string>();
        else if (m.is_object()) {
            if (m.count("id") && m["id"].is_string()) id = m["id"].get<std::string>();
            else if (m.count("name") && m["name"].is_string()) id = m["name"].get<std::string>();
            extract_pricing_from_model_obj(m, id);
        }
        if (!id.empty()) models.push_back(id);
    }
    std::vector<std::string> uniq;
    std::unordered_set<std::string> seen;
    for (auto& id : models) {
        if (!seen.count(id)) { seen.insert(id); uniq.push_back(id); }
    }
    // C3: hard cap in-memory list to protect low-RAM hosts
    if (uniq.size() > static_cast<size_t>(MODELS_MAX_CACHE)) {
        uniq.resize(static_cast<size_t>(MODELS_MAX_CACHE));
    }
    return uniq;
}

static bool refresh_models_from_api(bool force, bool quiet = false) {
    if (force) MODEL_PRICING_LIVE.clear();
    if (!force && load_models_cache()) {
        if (!quiet)
            std::cout << C_GRAY << "[models] cache: " << AVAILABLE_MODELS.size()
                      << " models" << C_RESET << std::endl;
        return true;
    }

    std::string api_key = get_api_key();
    if (api_key.empty()) {
        if (AVAILABLE_MODELS.empty()) AVAILABLE_MODELS = DEFAULT_MODELS;
        return false;
    }

    if (!quiet)
        std::cout << C_YELLOW << "[models] GET /v1/models ..." << C_RESET << std::endl;

    CURL *curl = curl_easy_init();
    if (!curl) return false;

    std::string response_body;
    struct curl_slist *headers = nullptr;
    std::string auth = "Authorization: Bearer " + api_key;
    headers = curl_slist_append(headers, auth.c_str());
    headers = curl_slist_append(headers, "Content-Type: application/json");

    std::string models_url = normalize_api_base(G.api_base) + "/v1/models";
    curl_easy_setopt(curl, CURLOPT_URL, models_url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 20L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        if (!quiet)
            std::cerr << C_RED << "[models] curl: " << curl_easy_strerror(res) << C_RESET << std::endl;
        if (AVAILABLE_MODELS.empty()) {
            if (!load_models_cache()) AVAILABLE_MODELS = DEFAULT_MODELS;
        }
        return false;
    }
    if (http_code != 200) {
        if (!quiet)
            std::cerr << C_RED << "[models] HTTP " << http_code << ": "
                      << response_body.substr(0, 200) << C_RESET << std::endl;
        if (AVAILABLE_MODELS.empty()) {
            if (!load_models_cache()) AVAILABLE_MODELS = DEFAULT_MODELS;
        }
        return false;
    }

    try {
        auto models = parse_models_json(response_body);
        if (models.empty()) {
            if (!quiet)
                std::cerr << C_RED << "[models] empty list from API" << C_RESET << std::endl;
            if (AVAILABLE_MODELS.empty()) AVAILABLE_MODELS = DEFAULT_MODELS;
            return false;
        }
        AVAILABLE_MODELS = models;
        save_models_cache(AVAILABLE_MODELS);
        if (!quiet)
            std::cout << C_GREEN << "[models] loaded: " << AVAILABLE_MODELS.size()
                      << " (cache: " << MODELS_CACHE_FILE << ")" << C_RESET << std::endl;
        return true;
    } catch (const std::exception& e) {
        if (!quiet)
            std::cerr << C_RED << "[models] parse: " << e.what() << C_RESET << std::endl;
        if (AVAILABLE_MODELS.empty()) AVAILABLE_MODELS = DEFAULT_MODELS;
        return false;
    }
}

static void print_models_list(const std::vector<std::string>& list,
                              bool show_header = true,
                              int page = 1,
                              const std::string& filter = "") {
    // /model N использует полный отфильтрованный список (не только текущую страницу)
    LAST_MODEL_VIEW = list;
    if (show_header) {
        std::cout << C_YELLOW << "\n[ models ]" << C_RESET;
        if (!filter.empty())
            std::cout << C_GRAY << " filter=\"" << filter << "\"" << C_RESET;
        std::cout << "\n";
    }
    if (list.empty()) {
        std::cout << C_GRAY << "  (пусто — смените фильтр или /models refresh)" << C_RESET << std::endl;
        return;
    }
    int page_sz = MODELS_PAGE_SIZE;
    int total = static_cast<int>(list.size());
    int pages = (total + page_sz - 1) / page_sz;
    if (pages < 1) pages = 1;
    if (page < 1) page = 1;
    if (page > pages) page = pages;
    int start = (page - 1) * page_sz;
    int end = std::min(total, start + page_sz);
    for (int i = start; i < end; ++i) {
        bool is_current = (list[i] == G.model);
        if (is_current) std::cout << C_GREEN << C_BOLD;
        else std::cout << C_CYAN;
        printf("  %3d) %s", i + 1, list[i].c_str());
        if (is_current) std::cout << "  <-- current";
        if (MODEL_PRICING_LIVE.count(list[i])) {
            auto pr = MODEL_PRICING_LIVE[list[i]];
            printf("  [$%.3g/$%.3g]", pr.first, pr.second);
        }
        std::cout << C_RESET << "\n";
    }
    std::cout << C_GRAY << "  shown: " << (end - start) << "/" << total
              << " | page " << page << "/" << pages
              << " | all_in_mem " << AVAILABLE_MODELS.size()
              << "/" << MODELS_MAX_CACHE
              << " | cache: " << MODELS_CACHE_FILE << C_RESET << std::endl;
    if (static_cast<int>(AVAILABLE_MODELS.size()) >= MODELS_MAX_CACHE)
        std::cout << C_YELLOW << "  [!] список обрезан cap=" << MODELS_MAX_CACHE
                  << " — уточните фильтр и /models refresh" << C_RESET << std::endl;
    std::cout << C_GRAY << "  выбрать: /model N  |  /model подстрока  |  /models слово1 слово2"
              << C_RESET << std::endl;
    if (pages > 1) {
        std::cout << C_GRAY << "  страницы: /models "
                  << (filter.empty() ? std::string("") : filter + " ")
                  << "p" << (page < pages ? page + 1 : 1);
        if (page > 1)
            std::cout << "  |  /models "
                      << (filter.empty() ? std::string("") : filter + " ")
                      << "p" << (page - 1);
        std::cout << "  |  /models refresh" << C_RESET << std::endl;
    }
}

[[maybe_unused]] static void print_models_list(bool show_header = true) {
    if (AVAILABLE_MODELS.empty())
        AVAILABLE_MODELS = DEFAULT_MODELS;
    print_models_list(AVAILABLE_MODELS, show_header, 1, "");
}

static void cmd_models(const std::string& arg) {
    std::string a = arg;
    while (!a.empty() && a[0] == ' ') a.erase(0, 1);
    while (!a.empty() && a.back() == ' ') a.pop_back();

    // tokens: refresh|cache|pN|filter words
    std::string filter;
    int page = 1;
    bool do_refresh = false, do_cache = false;
    {
        std::istringstream ss(a);
        std::string tok;
        while (ss >> tok) {
            std::string tl = to_lower_copy(tok);
            if (tl == "refresh" || tl == "live" || tl == "update" || tl == "force") do_refresh = true;
            else if (tl == "cache") do_cache = true;
            else if (tl.size() >= 2 && (tl[0] == 'p' || tl[0] == 'P') && std::isdigit(static_cast<unsigned char>(tl[1]))) {
                try { page = std::stoi(tl.substr(1)); } catch (...) {}
            } else if (tl == "page" || tl == "pg") {
                std::string n; if (ss >> n) { try { page = std::stoi(n); } catch (...) {} }
            } else {
                if (!filter.empty()) filter += " ";
                filter += tok;
            }
        }
    }

    if (do_refresh) {
        refresh_models_from_api(true, false);
    } else if (do_cache) {
        if (!load_models_cache()) {
            std::cout << C_YELLOW << "[models] empty cache, using defaults" << C_RESET << std::endl;
            AVAILABLE_MODELS = DEFAULT_MODELS;
        } else {
            std::cout << C_GRAY << "[models] from cache" << C_RESET << std::endl;
        }
    } else {
        if (!load_models_cache())
            refresh_models_from_api(true, false);
    }
    if (AVAILABLE_MODELS.empty())
        AVAILABLE_MODELS = DEFAULT_MODELS;

    auto list = filter_models(filter);
    print_models_list(list, true, page, filter);
}


void cmd_model_select() {
    if (AVAILABLE_MODELS.empty()) {
        if (!load_models_cache())
            refresh_models_from_api(true, false);
        if (AVAILABLE_MODELS.empty()) AVAILABLE_MODELS = DEFAULT_MODELS;
    }
    std::string filter;
    int page = 1;
    for (int step = 0; step < 8; ++step) {
        auto list = filter_models(filter);
        print_models_list(list, true, page, filter);
        std::cout << C_GRAY << "  сейчас: " << G.model << C_RESET << std::endl;
        char *rl_choice = readline(
            C_YELLOW "[N=номер | текст=фильтр | pN=страница | Enter=отмена]: " C_RESET);
        if (!rl_choice) return;
        std::string choice(rl_choice);
        free(rl_choice);
        while (!choice.empty() && (choice[0]==' '||choice[0]=='\t')) choice.erase(0,1);
        while (!choice.empty() && (choice.back()==' '||choice.back()=='\t')) choice.pop_back();
        if (choice.empty()) return;

        std::string cl = to_lower_copy(choice);
        if (cl.size() >= 2 && cl[0]=='p' && std::isdigit(static_cast<unsigned char>(cl[1]))) {
            try { page = std::stoi(cl.substr(1)); } catch (...) {}
            continue;
        }
        if (cl.rfind("page", 0) == 0) {
            try {
                auto sp = choice.find_first_of(" \t");
                if (sp != std::string::npos) page = std::stoi(choice.substr(sp + 1));
            } catch (...) {}
            continue;
        }
        bool pure_num = !choice.empty();
        for (unsigned char c : choice) if (!std::isdigit(c)) { pure_num = false; break; }
        if (pure_num) {
            std::string resolved, err;
            int rc = resolve_model_arg(choice, resolved, err);
            if (rc == 1) {
                G.model = resolved;
                save_config();
                std::cout << C_GREEN << "[Модель: " << G.model << "]" << C_RESET << std::endl;
                return;
            }
            std::cerr << C_RED << "[" << err << "]" << C_RESET << std::endl;
            continue;
        }
        std::string resolved, err;
        int rc = resolve_model_arg(choice, resolved, err);
        if (rc == 1) {
            G.model = resolved;
            save_config();
            std::cout << C_GREEN << "[Модель: " << G.model << "]" << C_RESET << std::endl;
            return;
        }
        filter = choice;
        page = 1;
        if (rc == 0)
            std::cout << C_YELLOW << "[" << err << " — показан фильтр]" << C_RESET << std::endl;
        else if (rc == 2)
            std::cout << C_YELLOW << "[" << err << "]" << C_RESET << std::endl;
    }
}

void print_help(bool full = false) {
    // D7: short by default, /help all for full
    if (!full) {
        std::cout << C_YELLOW
            << "Команды (кратко):\n"
            << "  /help all          — полная справка\n"
            << "  /model /models     — модель; /models claude sonnet | /model N|имя\n"
            << "  /apibase [url]     — API base (E3)\n"
            << "  /temp /maxtokens   — параметры генерации\n"
            << "  /file /save /load /history /clear /delete /retry\n"
            << "  /voice /speak /listen — голос (Termux API: STT+TTS)\n"
            << "  /autorun /nores /compact /dryrun /tokens /cost /balance\n"
            << "  /new /list /switch /rename /undo /alias /search /export\n"
            << "  /info N /dump [file]\n"
            << "  /update /about /exit\n"
            << "\nВвод: Enter/ '//' отправить | '.' пустая строка | Ctrl+C прервать запрос\n"
            << "Pipe: echo msg | sw_chat   |  sw_chat --exec  (D9 bash в pipe)\n"
            << C_RESET;
        return;
    }
    std::cout << C_YELLOW
        << "Специальные команды:\n"
        << "  /save              — сохранить историю\n"
        << "  /load              — загрузить историю\n"
        << "  /clear             — очистить историю диалога\n"
        << "  /history [on|off]  — показать историю / вкл-выкл сохранение\n"
        << "  /delete N          — удалить сообщение N из истории\n"
        << "  /retry             — повторить последний запрос (в т.ч. если модель не ответила)\n"
        << "  /tokens            — показать использование токенов\n"
        << "  /model [name|N]    — N из последнего /models; имя/фильтр; без args — меню\n"
        << "  /models [слова…] [pN|page N] [refresh|cache] — AND-фильтр; затем /model N\n"
        << "  /apibase [url]     — показать/задать API base (default 302.ai)\n"
        << "  /temp [0.0-2.0]    — показать/сменить температуру\n"
        << "  /maxtokens [N]     — показать/сменить max_tokens\n"
        << "  /system            — показать системный промпт\n"
        << "  /file <path> [msg] — загрузить файл и задать вопрос\n"
        << "  /voice             — вкл/выкл голосовой режим (инверсия)\n"
        << "  /voice on|off|in|out — вкл всё / выкл всё / только ввод / только озвучка\n"
        << "  /voice status      — показать настройки; /voice test [текст] — проверка звука\n"
        << "  /voice lang <code>|system, region <r>|none, variant <v>|none, engine <name>|default\n"
        << "  /voice stream <ALARM|MUSIC|NOTIFICATION|RING|SYSTEM|VOICE_CALL>\n"
        << "  /voice pitch <0.1..3.0>, rate <0.1..3.0>, short on|off (озвучивать 1-й абзац)\n"
        << "  (параметры TTS; распознавание речи язык берёт из настроек Android)\n"
        << "  /speak <текст>     — произнести текст вслух (termux-tts-speak)\n"
        << "  /listen            — распознать речь и отправить как сообщение\n"
        << "  /autorun           — вкл/выкл авто-выполнение bash\n"
        << "  /nores             — вкл/выкл вывод результатов bash\n"
        << "  /time on|off       — добавлять дату-время (МСК) в начало каждого запроса к модели\n"
        << "                       (модель видит дату-время; на экран ничего не выводится)\n"
        << "  /FIRE [on|off|blast] — тихий bash: скрыть код+вывод+bash-блок, autorun вкл\n"
        << "                       on=вкл; off=выкл (autorun тоже выкл); blast — аварийный выход\n"
        << "  /dryrun            — вкл/выкл dry-run bash-блоков (не выполнять, только показывать)\n"
        << "  /compact           — тихий режим (plain, без подсказок/spinner)\n"
        << "  /cost [live]       — стоимость токенов ($); live — подтянуть цены API\n"
        << "  /balance           — ключ/провайдер и токены сессии\n"
        << "  /update            — обновление программы (+ preview изменений)\n"
        << "  /about             — информация о программе\n"
        << "  /new [name]        — создать новую сессию\n"
        << "  /list              — список сессий\n"
        << "  /switch <name>     — переключить сессию\n"
        << "  /rename <name>     — переименовать текущую сессию\n"
        << "  /undo              — откатить последний шаг (запрос)\n"
        << "  /alias k=v         — создать/удалить/показать алиасы\n"
        << "  /search <text>     — поиск по истории\n"
        << "  /export [fmt] [f]  — экспорт диалога (md/txt/json), по умолчанию ~/tmp/dialog_export.<fmt>\n"
        << "  /info N            — показать сообщение N полностью\n"
        << "  /dump [file]       — сохранить последний вывод bash (~/tmp/last_bash.txt)\n"
        << "  /help [all]        — краткая / полная справка\n"
        << "  /exit              — выход\n"
        << "\nМногострочный ввод:\n"
        << "  Пустой Enter      — отправить сообщение\n"
        << "  //                 — отправить сообщение (конец ввода)\n"
        << "  .                  — вставить пустую строку\n"
        << "\nPipe / args:\n"
        << "  sw_chat --exec ... — выполнить bash-блоки из ответа (D9)\n"
        << "  sw_chat --dry-run ... — показать, какие команды были бы выполнены (не выполнять)\n"
        << "\nВо время получения ответа:\n"
        << "  Ctrl+C             — прервать запрос\n"
        << "  bash chain N/7     — счётчик последовательных запросов к модели (D8)\n"
        << C_RESET;
}

void print_history() {
    std::cout << C_YELLOW << "[История диалога (" << G.messages.size()
              << " сообщений)]:" << C_RESET << std::endl;
    for (size_t i = 0; i < G.messages.size(); ++i) {
        std::string role = G.messages[i].value("role", std::string("?"));
        std::string cont;
        if (G.messages[i].count("content") && G.messages[i]["content"].is_string())
            cont = G.messages[i]["content"].get<std::string>();
        else if (G.messages[i].count("content"))
            cont = G.messages[i]["content"].dump();
        if (cont.size() > 120) cont = cont.substr(0, 120) + "...";
        if (role == "system")
            std::cout << C_MAGENTA << "[" << i << "] system: "    << C_RESET << cont << "\n";
        else if (role == "user")
            std::cout << C_GREEN   << "[" << i << "] user: "      << C_RESET << cont << "\n";
        else if (role == "assistant")
            std::cout << C_CYAN    << "[" << i << "] assistant: " << C_RESET << cont << "\n";
        else
            std::cout << C_YELLOW  << "[" << i << "] " << role << ": " << C_RESET << cont << "\n";
    }
}

// /info N — показать сообщение N полностью (в отличие от /history)
void cmd_info(const std::string& arg) {
    if (arg.empty()) {
        std::cerr << C_RED << "[Использование: /info N]" << C_RESET << std::endl;
        return;
    }
    try {
        int idx = std::stoi(arg);
        if (idx < 0 || idx >= static_cast<int>(G.messages.size())) {
            std::cerr << C_RED << "[Неверный индекс: " << idx
                      << " (всего " << G.messages.size() << ")]" << C_RESET << std::endl;
            return;
        }
        std::string role = G.messages[idx].value("role", std::string("?"));
        std::string cont;
        if (G.messages[idx].count("content") && G.messages[idx]["content"].is_string())
            cont = G.messages[idx]["content"].get<std::string>();
        else if (G.messages[idx].count("content"))
            cont = G.messages[idx]["content"].dump();
        const char* col = (role == "system")    ? C_MAGENTA
                        : (role == "user")      ? C_GREEN
                        : (role == "assistant") ? C_CYAN
                                                : C_YELLOW;
        std::cout << col << "[" << idx << "] " << role
                  << " (" << cont.size() << " байт):" << C_RESET << "\n";
        render_markdown(cont);
        std::cout << std::endl;
    } catch (...) {
        std::cerr << C_RED << "[Использование: /info N]" << C_RESET << std::endl;
    }
}

// /dump [файл] — сохранить последний вывод bash
void cmd_dump(const std::string& arg) {
    if (g_last_bash_result.empty() && g_last_bash_code.empty()) {
        std::cout << C_GRAY << "[Нет сохранённого bash-вывода для дампа]" << C_RESET << std::endl;
        return;
    }
    std::string file = arg;
    if (file.empty())
        file = get_home_dir() + "/tmp/last_bash.txt";
    if (!file.empty() && file[0] == '~')
        file = get_home_dir() + file.substr(1);
    std::ofstream f(file);
    if (!f.is_open()) {
        std::cerr << C_RED << "[Не удалось открыть " << file << "]" << C_RESET << std::endl;
        return;
    }
    f << "# command:\n" << g_last_bash_code << "\n\n# output:\n" << g_last_bash_result;
    std::cout << C_GREEN << "[Дамп сохранён: " << file << " ("
              << g_last_bash_result.size() << " байт)]" << C_RESET << std::endl;
}

void print_tokens() {
    std::cout << C_MAGENTA
              << "[Токены — промпт: " << G.total_prompt_tokens
              << ", ответы: "         << G.total_completion_tokens
              << ", итого: "          << (G.total_prompt_tokens + G.total_completion_tokens)
              << "]" << C_RESET << std::endl;
}

void cmd_delete(const std::string &arg) {
    if (arg.empty()) {
        std::cerr << C_RED << "[Использование: /delete N]" << C_RESET << std::endl;
        return;
    }
    try {
        int idx = std::stoi(arg);
        if (idx < 0 || idx >= static_cast<int>(G.messages.size())) {
            std::cerr << C_RED << "[Неверный индекс: " << idx << "]" << C_RESET << std::endl;
            return;
        }
        if (G.messages[idx].value("role", "") == "system") {
            std::cerr << C_RED << "[Нельзя удалить системный промпт]" << C_RESET << std::endl;
            return;
        }
        G.messages.erase(G.messages.begin() + idx);
        std::cout << C_YELLOW << "[Сообщение " << idx << " удалено]" << C_RESET << std::endl;
    } catch (...) {
        std::cerr << C_RED << "[Использование: /delete N]" << C_RESET << std::endl;
    }
}

// ─────────────────────────── /file ───────────────────────────
void cmd_file(const std::string &arg) {
    if (arg.empty()) {
        std::cerr << C_RED << "[Использование: /file <путь> [вопрос]]" << C_RESET << std::endl;
        return;
    }
    // Разделяем путь и опциональный вопрос
    std::string path, question;
    // Если путь в кавычках
    if (arg[0] == '"' || arg[0] == '\'') {
        char quote = arg[0];
        size_t end = arg.find(quote, 1);
        if (end != std::string::npos) {
            path = arg.substr(1, end - 1);
            if (end + 2 < arg.size()) question = arg.substr(end + 2);
        } else {
            path = arg.substr(1);
        }
    } else {
        size_t sp = arg.find(' ');
        if (sp != std::string::npos) {
            path = arg.substr(0, sp);
            question = arg.substr(sp + 1);
        } else {
            path = arg;
        }
    }
    // Раскрываем ~ в начале пути
    if (!path.empty() && path[0] == '~') {
        path = get_home_dir() + path.substr(1);
    }
    struct stat st{};
    if (stat(path.c_str(), &st) != 0) {
        std::cerr << C_RED << "[Не удалось открыть файл: " << path << "]" << C_RESET << std::endl;
        return;
    }
    if (!S_ISREG(st.st_mode)) {
        std::cerr << C_RED << "[Не обычный файл: " << path << "]" << C_RESET << std::endl;
        return;
    }
    if (st.st_size > static_cast<off_t>(MAX_FILE_BYTES)) {
        std::cerr << C_RED << "[Файл слишком большой: " << st.st_size
                  << " байт (лимит " << MAX_FILE_BYTES << ")]" << C_RESET << std::endl;
        return;
    }
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << C_RED << "[Не удалось открыть файл: " << path << "]" << C_RESET << std::endl;
        return;
    }
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    if (content.empty()) {
        std::cerr << C_RED << "[Файл пуст: " << path << "]" << C_RESET << std::endl;
        return;
    }
    if (content.size() > static_cast<size_t>(MAX_FILE_BYTES)) {
        content = content.substr(0, static_cast<size_t>(MAX_FILE_BYTES));
        size_t cut = content.size();
        while (cut > 0 && (content[cut-1] & 0xC0) == 0x80) --cut;
        content.resize(cut);
        content += "\n[...файл обрезан по лимиту " + std::to_string(MAX_FILE_BYTES) + " байт...]";
        std::cout << C_YELLOW << "[Файл обрезан до " << MAX_FILE_BYTES << " байт]" << C_RESET << std::endl;
    }
    // Определяем расширение для подсветки
    std::string ext;
    size_t dot = path.rfind('.');
    if (dot != std::string::npos) ext = path.substr(dot + 1);
    // Формируем сообщение
    std::string msg = "Файл `" + path + "` (" + std::to_string(content.size()) + " байт):\n```" + ext + "\n" + content;
    // Закрываем блок кода если нет завершающего newline
    if (!msg.empty() && msg.back() != '\n') msg += "\n";
    msg += "```";
    if (!question.empty()) {
        msg += "\n\n" + question;
    }
    std::cout << C_YELLOW << "[Файл загружен: " << path << " ("
              << content.size() << " байт)]" << C_RESET << std::endl;
    push_undo_snapshot();
    G.messages.push_back({{"role", "user"}, {"content", msg}});
    // Делаем запрос к API
    size_t msgs_before = G.messages.size() - 1;
    bool aborted = false;
    std::string response = do_api_request(aborted);
    if (g_exit_requested) do_exit();
    process_response(response, aborted, msgs_before);
}

// ─────────────────────────── /cost ───────────────────────────
struct ModelPricing {
    const char* model_prefix;
    double prompt_per_mtok;     // $ per 1M prompt tokens
    double completion_per_mtok; // $ per 1M completion tokens
};

static const ModelPricing KNOWN_PRICING[] = {
    {"anthropic/claude-opus-4.8",       5.0,   25.0},
    {"anthropic/claude-sonnet-4.6",     3.0,   15.0},
    {"anthropic/claude-haiku",          1.0,    5.0},
    {"openai/gpt-5",                    2.5,   15.0},
    {"openai/gpt-4.1",                  2.0,    8.0},
    {"openai/gpt-4.1-mini",             0.4,    1.6},
    {"openai/gpt-4.1-nano",             0.1,    0.4},
    {"openai/o3",                       2.0,    8.0},
    {"openai/o4-mini",                  1.1,    4.4},
    {"google/gemini-2.5-pro",           1.25,  10.0},
    {"google/gemini-2.5-flash",         0.3,    2.5},
    {"google/gemini-3",                 2.0,   12.0},
    {"x-ai/grok-4",                     3.0,   15.0},
    {"x-ai/grok-3",                     3.0,   15.0},
    {"x-ai/grok-3-mini",                0.3,    0.5},
    {"deepseek/deepseek-r1",            0.7,    2.5},
    {"deepseek/deepseek-chat",          0.26,  0.38},
    {"deepseek-chat",                   0.26,  0.38},
    {"deepseek-reasoner",               0.55,  2.19},
    {"deepseek/deepseek-v4-pro",        0.435, 0.87},
    {"qwen/qwen3",                      0.39,  2.34},
    {"meta-llama/llama-4",              0.15,   0.6},
    {"minimax/minimax-m2.7",            0.3,    1.2},
    {"xiaomi/mimo-v2-pro",              1.0,    3.0},
    {"~google/gemini-pro-latest",       1.05,   2.0},
    {"~anthropic/claude-sonnet-latest", 1.0,    3.0},
    {"qwen/qwen3.6-max-preview",        1.04,  6.24},
    {nullptr, 0, 0}
};

void print_cost(bool live = false) {
    if (live) {
        std::cout << C_YELLOW << "[cost] refresh models/pricing from API..." << C_RESET << std::endl;
        auto pricing_backup = MODEL_PRICING_LIVE;
        refresh_models_from_api(true, false);
        if (MODEL_PRICING_LIVE.empty())
            MODEL_PRICING_LIVE = pricing_backup; // сеть упала — восстановить прежние цены
    } else if (MODEL_PRICING_LIVE.empty()) {
        load_models_cache(); // may fill pricing from cache
    }

    double p_price = 0, c_price = 0;
    bool found = false;
    std::string price_src;

    // 1) live map exact / substring
    if (MODEL_PRICING_LIVE.count(G.model)) {
        p_price = MODEL_PRICING_LIVE[G.model].first;
        c_price = MODEL_PRICING_LIVE[G.model].second;
        found = true;
        price_src = "api/cache";
    } else {
        for (auto& kv : MODEL_PRICING_LIVE) {
            if (G.model.find(kv.first) != std::string::npos || kv.first.find(G.model) != std::string::npos) {
                p_price = kv.second.first; c_price = kv.second.second;
                found = true; price_src = "api/cache~" + kv.first;
                break;
            }
        }
    }
    // 2) static table
    if (!found) {
        for (int i = 0; KNOWN_PRICING[i].model_prefix != nullptr; ++i) {
            if (G.model.find(KNOWN_PRICING[i].model_prefix) == 0 ||
                std::string(KNOWN_PRICING[i].model_prefix).find(G.model) != std::string::npos ||
                G.model == KNOWN_PRICING[i].model_prefix) {
                p_price = KNOWN_PRICING[i].prompt_per_mtok;
                c_price = KNOWN_PRICING[i].completion_per_mtok;
                found = true;
                price_src = "builtin";
                break;
            }
        }
    }

    double prompt_cost = (G.total_prompt_tokens / 1000000.0) * p_price;
    double completion_cost = (G.total_completion_tokens / 1000000.0) * c_price;
    double total_cost = prompt_cost + completion_cost;
    int total_tokens = G.total_prompt_tokens + G.total_completion_tokens;
    size_t est_now = approx_tokens_messages();

    std::cout << C_MAGENTA << "\n  Использование токенов" << C_RESET << "\n";
    std::cout << C_GRAY << "  ────────────────────────────────" << C_RESET << "\n";
    std::cout << C_MAGENTA << "  Модель:\t" << C_RESET << G.model << "\n";
    std::cout << C_MAGENTA << "  API:\t\t" << C_RESET << normalize_api_base(G.api_base) << "\n";
    std::cout << C_MAGENTA << "  Промпт:\t" << C_RESET << G.total_prompt_tokens << " токенов\n";
    std::cout << C_MAGENTA << "  Ответы:\t" << C_RESET << G.total_completion_tokens << " токенов\n";
    std::cout << C_MAGENTA << "  Всего:\t" << C_RESET << total_tokens << " токенов\n";
    std::cout << C_MAGENTA << "  Контекст:~\t" << C_RESET << est_now << " tok (est chars/4)\n";
    if (found) {
        std::cout << C_GRAY << "  ────────────────────────────────" << C_RESET << "\n";
        printf("  Промпт:\t$%.4f\n", prompt_cost);
        printf("  Ответы:\t$%.4f\n", completion_cost);
        printf("  Итого:\t$%.4f\n", total_cost);
        printf("  ($%.4g/$%.4g за 1M токенов, src=%s)\n", p_price, c_price, price_src.c_str());
    } else {
        std::cout << C_GRAY << "\n  Цены для модели не найдены. Попробуйте: /cost live" << C_RESET << "\n";
    }
    std::cout << std::endl;
}

// ─────────────────────────── Ввод пользователя ───────────────
static std::string g_rl_prefill;
// FIX(BUG2): если STT несколько раз подряд дал пустой результат
// (типичный случай: движок распознавания не установлен —
//  logcat onError(13)==ERROR_LANGUAGE_UNAVAILABLE), предупреждаем
// и авто-выключаем voice_in, чтобы не мучить пользователя.
static int   g_stt_empty_streak = 0;
static bool  g_stt_unavailable_notified = false;
static const int STT_EMPTY_LIMIT = 3;
static int voice_prefill_hook() {
    if (!g_rl_prefill.empty()) rl_insert_text(g_rl_prefill.c_str());
    return 0;
}

static bool get_user_input(std::string &out) {
    std::string result;
    bool first_line = true;
    // bool multiline = false; // БАГ 7: переменная объявлялась но нигде не читалась
    int  line_num   = 1;

    while (true) {
        if (g_exit_requested) return false;

        std::string prompt = first_line
            ? build_prompt()
            : ("\001" C_GREEN "\002" + std::to_string(line_num) + "\xe2\x80\xa6 \001" C_INPUT "\002");

        // Голосовой ввод: распознаём речь → подставляем текст в readline
        // (можно отредактировать или сразу нажать Enter).
        if (first_line && G.voice_in && have_termux_api("termux-speech-to-text")) {
            std::cout << C_YELLOW << "[voice: слушаю... (Ctrl+C — отмена)]" << C_RESET << std::endl;
            std::cout.flush();
            std::string rec = voice_recognize();
            if (rec.empty()) {
                ++g_stt_empty_streak;
                if (g_stt_empty_streak >= STT_EMPTY_LIMIT && !g_stt_unavailable_notified) {
                    std::cout << C_RED << "[voice: STT-движок, похоже, недоступен на устройстве "
                              << "(" << g_stt_empty_streak << " пустых подряд). "
                              << "Выключаю voice_in (/voice на — включить обратно)]" << C_RESET << std::endl;
                    G.voice_in = false;
                    save_config();
                    g_stt_unavailable_notified = true;
                } else {
                    std::cout << C_GRAY << "[voice: ничего не распознано]"
                              << (g_stt_empty_streak > 1
                                      ? " (" + std::to_string(g_stt_empty_streak) + "/"
                                            + std::to_string(STT_EMPTY_LIMIT) + ")"
                                      : std::string())
                              << C_RESET << std::endl;
                }
            } else {
                g_stt_empty_streak = 0;
                std::cout << C_CYAN << "[voice]: " << rec << C_RESET << std::endl;
                g_rl_prefill = rec;
            }
        }

        std::cout.flush(); fflush(stdout); // Сброс буферов перед readline
        rl_startup_hook = voice_prefill_hook;
        char *line = readline(prompt.c_str());
        std::cout << C_RESET;  // сброс цвета ввода пользователя
        rl_startup_hook = nullptr;
        g_rl_prefill.clear();

        if (!line) {
            // EOF (Ctrl+D)
            if (!result.empty()) {
                out = result;
                if (G.history_enabled) add_history(history_oneline(result).c_str());
                return true;
            }
            return false;
        }

        std::string sline = sanitize_utf8(std::string(line));
        if (line) free(line);

        // "//" — завершение многострочного ввода
        if (sline == "//") {
            if (result.empty()) {
                std::cout << C_GRAY
                          << "[Нет текста для отправки]"
                          << C_RESET << std::endl;
                first_line = true;
                line_num   = 1;
                result.clear();
                continue;
            }
            break;
        }

        if (first_line) {
            if (sline.empty()) {
                // Пустая строка на первой позиции — пустой ввод
                out = "";
                return true;
            }
            // Первая непустая строка — добавляем и переходим в многострочный режим
            result = sline;
            first_line = false;
            line_num   = 2;
            if (!is_compact()) {
            std::cout << C_GRAY
                      << "[Многострочный режим: пустой Enter — отправить, '.' — пустая строка, '//' — отправить]"
                      << C_RESET << std::endl;
            }
        } else {
            // Пустая строка в многострочном режиме — отправляем
            if (sline.empty()) {
                break;
            }
            // Одиночная точка — вставить пустую строку
            if (sline == ".") {
                result += "\n";
            } else {
                result += "\n";
                result += sline;
            }
            line_num++;
        }
    }

    if (!result.empty()) {
        if (G.history_enabled) {
            add_history(history_oneline(result).c_str());
        }
    }

    out = result;
    return true;
}

// ─────────────────────────── Команды ─────────────────────────
static bool match_command(const std::string &s, const std::string &cmd) {
    if (s.size() < cmd.size()) return false;
    if (s.substr(0, cmd.size()) != cmd) return false;
    if (s.size() == cmd.size()) return true;
    return s[cmd.size()] == ' ';
}

static std::string command_arg(const std::string &s, const std::string &cmd) {
    if (s.size() <= cmd.size() + 1) return "";
    return s.substr(cmd.size() + 1);
}

// ─────────────────────────── /voice ──────────────────────────
static void cmd_voice(const std::string& arg) {
    std::string a = arg;
    while (!a.empty() && a[0]==' ') a.erase(0,1);
    while (!a.empty() && a.back()==' ') a.pop_back();
    std::vector<std::string> tok;
    { std::istringstream ss(a); std::string t; while (ss >> t) tok.push_back(t); }

    auto val = [&]() -> std::string {
        std::string v;
        for (size_t i=1;i<tok.size();++i) { if (!v.empty()) v+=" "; v+=tok[i]; }
        return v;
    };
    auto show_status = [&]() {
        std::string lang = G.voice_lang.empty() ? std::string("(system)") : G.voice_lang;
        std::string eng  = G.voice_engine.empty() ? std::string("(default)") : G.voice_engine;
        char b[64];
        std::cout << C_GREEN << "[voice: ввод " << (G.voice_in?"ON":"off")
                  << ", вывод " << (G.voice_out?"ON":"off") << "]" << C_RESET;
        std::cout << C_GRAY << " lang=" << lang;
        if (!G.voice_region.empty())  std::cout << " region=" << G.voice_region;
        if (!G.voice_variant.empty()) std::cout << " variant=" << G.voice_variant;
        snprintf(b, sizeof(b), " pitch=%.2f", G.voice_pitch); std::cout << b;
        snprintf(b, sizeof(b), " rate=%.2f",  G.voice_rate);  std::cout << b;
        std::cout << " stream=" << G.voice_stream
                  << " engine=" << eng
                  << " short=" << (G.voice_short?"on":"off")
                  << C_RESET << std::endl;
    };

    if (tok.empty()) {
        bool on = !(G.voice_in || G.voice_out);
        G.voice_in = on; G.voice_out = on;
        save_config();
        std::cout << C_GREEN << "[voice: ввод " << (G.voice_in?"ON":"off")
                  << ", вывод " << (G.voice_out?"ON":"off") << "]" << C_RESET << std::endl;
        return;
    }
    const std::string k = tok[0];
    if (k=="on")  { G.voice_in=true;  G.voice_out=true;  save_config();
                    g_stt_empty_streak = 0; g_stt_unavailable_notified = false;
                    std::cout<<"[voice: ON]"<<std::endl;  return; }
    if (k=="off") { G.voice_in=false; G.voice_out=false; save_config(); std::cout<<"[voice: off]"<<std::endl; return; }
    if (k=="in")  { G.voice_in=true;  G.voice_out=false; save_config();
                    g_stt_empty_streak = 0; g_stt_unavailable_notified = false;
                    std::cout<<"[voice: только ввод]"<<std::endl;  return; }
    if (k=="out") { G.voice_in=false; G.voice_out=true;  save_config(); std::cout<<"[voice: только озвучка]"<<std::endl; return; }
    if (k=="status") { show_status(); return; }
    if (k=="test") {
        std::string v = val(); if (v.empty()) v = "Это тест голосового вывода.";
        voice_speak(v);
        std::cout << "[voice test: " << v << "]" << std::endl;
        return;
    }
    if (k=="lang") {
        std::string v = val();
        if (v.empty()) { std::cout << "[lang: " << (G.voice_lang.empty()?"(system)":G.voice_lang) << "]" << std::endl; return; }
        if (v=="system"||v=="auto"||v=="-") G.voice_lang.clear(); else G.voice_lang=v;
        save_config(); std::cout << "[voice lang: " << (G.voice_lang.empty()?"(system)":G.voice_lang) << "]" << std::endl; return;
    }
    if (k=="region") {
        std::string v = val();
        if (v.empty()) { std::cout << "[region: " << (G.voice_region.empty()?"(none)":G.voice_region) << "]" << std::endl; return; }
        if (v=="none"||v=="-") G.voice_region.clear(); else G.voice_region=v;
        save_config(); std::cout << "[voice region: " << (G.voice_region.empty()?"(none)":G.voice_region) << "]" << std::endl; return;
    }
    if (k=="variant") {
        std::string v = val();
        if (v.empty()) { std::cout << "[variant: " << (G.voice_variant.empty()?"(none)":G.voice_variant) << "]" << std::endl; return; }
        if (v=="none"||v=="-") G.voice_variant.clear(); else G.voice_variant=v;
        save_config(); std::cout << "[voice variant: " << (G.voice_variant.empty()?"(none)":G.voice_variant) << "]" << std::endl; return;
    }
    if (k=="engine") {
        std::string v = val();
        if (v.empty()) { std::cout << "[engine: " << (G.voice_engine.empty()?"(default)":G.voice_engine) << "]" << std::endl; return; }
        if (v=="default"||v=="-") G.voice_engine.clear(); else G.voice_engine=v;
        save_config(); std::cout << "[voice engine: " << (G.voice_engine.empty()?"(default)":G.voice_engine) << "]" << std::endl; return;
    }
    if (k=="pitch") {
        std::string v = val();
        if (v.empty()) { printf("[pitch: %.2f]\n", G.voice_pitch); return; }
        try { double d=std::stod(v);
            if (d<0.1||d>3.0) { std::cout<<C_RED<<"[pitch: диапазон 0.1..3.0]"<<C_RESET<<std::endl; return; }
            G.voice_pitch=d; save_config(); printf("[voice pitch: %.2f]\n", G.voice_pitch);
        } catch(...) { std::cout<<C_RED<<"[pitch: неверное число]"<<C_RESET<<std::endl; }
        return;
    }
    if (k=="rate") {
        std::string v = val();
        if (v.empty()) { printf("[rate: %.2f]\n", G.voice_rate); return; }
        try { double d=std::stod(v);
            if (d<0.1||d>3.0) { std::cout<<C_RED<<"[rate: диапазон 0.1..3.0]"<<C_RESET<<std::endl; return; }
            G.voice_rate=d; save_config(); printf("[voice rate: %.2f]\n", G.voice_rate);
        } catch(...) { std::cout<<C_RED<<"[rate: неверное число]"<<C_RESET<<std::endl; }
        return;
    }
    if (k=="stream") {
        std::string v = val();
        if (v.empty()) { std::cout << "[stream: " << G.voice_stream << "]" << std::endl; return; }
        std::string V=v; for (auto& c:V) c=static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        const char* ok[]={"ALARM","MUSIC","NOTIFICATION","RING","SYSTEM","VOICE_CALL",nullptr};
        bool good=false; for (int i=0; ok[i]; ++i) if (V==ok[i]) { good=true; break; }
        if (!good) { std::cout<<C_RED<<"[stream: ALARM|MUSIC|NOTIFICATION|RING|SYSTEM|VOICE_CALL]"<<C_RESET<<std::endl; return; }
        G.voice_stream=V; save_config();
        std::cout << "[voice stream: " << G.voice_stream << "]" << std::endl; return;
    }
    if (k=="short") {
        std::string v = val();
        if (v.empty()) { std::cout << "[short: " << (G.voice_short?"on":"off") << "]" << std::endl; return; }
        std::string vl=v; for (auto& c:vl) c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (vl=="on"||vl=="y"||vl=="yes") { G.voice_short=true;  save_config(); std::cout<<"[voice short: on]"<<std::endl; }
        else if (vl=="off"||vl=="n"||vl=="no") { G.voice_short=false; save_config(); std::cout<<"[voice short: off]"<<std::endl; }
        else std::cout<<C_RED<<"[short: on|off]"<<C_RESET<<std::endl;
        return;
    }
    std::cout << C_YELLOW
        << "Использование: /voice [on|off|in|out|status|test [текст]]\n"
        << "  параметры: lang <code> | region <r> | variant <v> | engine <name>\n"
        << "             stream <ALARM|MUSIC|NOTIFICATION|RING|SYSTEM|VOICE_CALL>\n"
        << "             pitch <0.1..3.0> | rate <0.1..3.0> | short on|off"
        << C_RESET << std::endl;
}

// ─────────────────────────── main ────────────────────────────
int main(int argc, char *argv[]) {

    std::setlocale(LC_ALL, "");
    {
        bool want_help = false, want_ver = false, full_help = false;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--version" || a == "-v") want_ver = true;
            else if (a == "--help" || a == "-h") want_help = true;
            else if (want_help && (a == "all" || a == "full" || a == "a")) full_help = true;
        }
        if (want_ver) { std::cout << "sw_chat " << APP_VERSION << "\n"; return 0; }
        if (want_help) { print_help(full_help); return 0; }
    }
    rl_catch_signals  = 0;   // Мы сами обрабатываем сигналы
    rl_catch_sigwinch = 1;   // Readline сам обрабатывает ресайз окна
    // Unified path/config init for interactive and pipe/args modes
    init_paths();
    load_config();
    // session_name мог быть прочитан из конфига — пересчитать пути сессии
    HISTORY_FILE = SESSIONS_DIR + "/" + G.session_name + ".json";
    G.history_file = HISTORY_FILE;
    SYSTEM_PROMPT_FILE = get_home_dir() + "/tmp/system_prompt.txt";


    {
        struct sigaction sa;
        std::memset(&sa, 0, sizeof(sa));
        sa.sa_handler = signal_handler;
        sigemptyset(&sa.sa_mask);
        // ВАЖНО: без SA_RESTART — блокирующие syscall'ы (read/poll/recv)
        // возвращают EINTR, что позволяет мгновенно реагировать на Ctrl+C.
        sa.sa_flags = 0;
        sigaction(SIGINT,  &sa, nullptr);
        sigaction(SIGTERM, &sa, nullptr);
    }

    G.sys_prompt = load_system_prompt();
    if (G.sys_prompt.empty()) {
        G.sys_prompt =
            "FIRE-режим (/FIRE on): bash-блоки выполняются АВТОМАТИЧЕСКИ и МОЛЧА — пользователь НЕ видит ни команду, ни результат. "
            "В FIRE категорически избегай деструктивных и рискованных команд (rm -rf, mkfs, dd of=, перезапись /dev/*, fork-bomb, пайпы в sh/bash, chmod -R 777, git push --force): они могут необратимо что-то испортить без ведома пользователя. "
            "Если такая команда действительно нужна — не выполняй её в FIRE: сначала напиши пользователю обычным текстом, что требуется /FIRE off и его явное согласие. "
            "То, что ты выведешь после ```bash будет сразу исполняться в системе через функцию system();. "
            "Используй максимально аккуратно, чтобы не навредить системе !!! "
            "Всегда придерживайся правила: несколько bash-блоков могут быть в твоём ответе, все будут выполнены последовательно. "
            "При выводе тобой bash-блока ничего больше не выводить, пока я разрешу или не разрешу. "
            "Все инструкции, что указаны здесь выше ты должен постоянно помнить и не нарушать. "
            "ЭТО ВАЖНО! Результат выполнения команды будет добавлен к твоему сообщению автоматически. "
            "В САМОМ НАЧАЛЕ запросов пользователя может стоять строка вида "
            "[Текущие дата и время (МСК, UTC+3): дд.мм.гггг чч:мм:сс] — это реальное текущее время пользователя. "
            "Используй его для ориентации во времени: паузы между сообщениями, время суток у пользователя, "
            "планирование взаимодействия в реальном времени и т.п. "
            "В папке ~/tmp возможно будет файл memo.md это твоя память. "
            "Если необходимо сделать запись в memo.md, то сохраняй самое важное, максимум три - пять строк, ДОПИСЫВАЯ в файл.";
        // Автосоздание редактируемого файла системного промпта (для правки).
        {
            std::ofstream sp_out(SYSTEM_PROMPT_FILE);
            if (sp_out.is_open()) {
                sp_out << G.sys_prompt;
                sp_out.close();
                if (!G.fire)
                    std::cout << C_GRAY << "[Создан системный промпт: " << SYSTEM_PROMPT_FILE
                              << " — можно редактировать]" << C_RESET << std::endl;
            }
        }
    }
    G.messages.push_back({{"role", "system"}, {"content", G.sys_prompt}});

    curl_global_init(CURL_GLOBAL_ALL);

    // ── Проверка --restore-session после обновления ──
    bool restore_session = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--restore-session") {
            restore_session = true;
            break;
        }
    }
    if (restore_session) {
        load_history();
        std::cout << C_GREEN << "[update] Сессия восстановлена (" << G.messages.size() - 1 << " сообщений)" << C_RESET << std::endl;
    }

    // ── Режим пайпа / аргументов ──
    bool pipe_mode = !isatty(fileno(stdin));
    bool exec_mode = false; // D9: run bash blocks from response
    int real_args = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--restore-session") continue;
        if (a == "--exec" || a == "-e") { exec_mode = true; continue; }
        if (a == "--dry-run" || a == "-n") { g_dry_run = true; continue; }
        real_args++;
    }
    bool has_args  = real_args > 0;

    if (pipe_mode || has_args) {
        std::string message;
        if (has_args) {
            for (int i = 1; i < argc; ++i) {
                std::string a = argv[i];
                if (a == "--restore-session" || a == "--exec" || a == "-e" ||
                    a == "--dry-run" || a == "-n") continue;
                if (!message.empty()) message += " ";
                message += a;
            }
        }
        if (pipe_mode) {
            std::string pipe_data, pline;
            while (std::getline(std::cin, pline)) {
                if (!pipe_data.empty()) pipe_data += "\n";
                pipe_data += pline;
            }
            if (!pipe_data.empty()) {
                if (!message.empty()) message += "\n\n";
                message += pipe_data;
            }
        }
        if (message.empty()) {
            std::cerr << C_RED << "[Нет данных]" << C_RESET << std::endl;
            curl_global_cleanup();
            return 1;
        }
        // In non-tty exec mode default compact-ish confirmations still apply unless autorun
        G.messages.push_back({{"role", "user"}, {"content", clamp_message_content(message)}});
        size_t msgs_before = G.messages.size() - 1;
        bool aborted = false;
        std::string content;
        try {
            content = do_api_request(aborted);
        } catch (const std::exception& e) {
            std::cerr << C_RED << "[Ошибка API: " << e.what() << "]" << C_RESET << std::endl;
            content = "";
        } catch (...) {
            std::cerr << C_RED << "[Неизвестная ошибка API]" << C_RESET << std::endl;
            content = "";
        }
        if (!content.empty()) {
            if (exec_mode) {
                // D9: full bash pipeline processing
                process_response(content, aborted, msgs_before);
            } else {
                print_assistant_text(content, false);
            }
        }
        curl_global_cleanup();
        return 0;
    }

    // ── Интерактивный режим ──

    if (is_compact()) {
        std::cout << G.model << std::endl;
    } else {
    std::cout << C_BOLD << C_CYAN << "=== Chat CLI ===" << C_RESET << std::endl;
    std::cout << C_YELLOW << "Модель: " << G.model << C_RESET << std::endl;
    std::cout << C_YELLOW << "Введите /help для справки" << C_RESET << std::endl;
    std::cout << C_GRAY   << "Autorun: " << C_RESET
              << (G.autorun ? C_RED "ВКЛ ⚠" : C_GRAY "выкл") << C_RESET
              << C_GRAY << " (переключить: /autorun)" << C_RESET
              << C_GRAY << " FIRE: " << C_RESET
              << (G.fire ? C_RED "ВКЛ 🔥" : C_GRAY "выкл") << C_RESET
              << C_GRAY << " (/FIRE)" << C_RESET
              << C_GRAY << " Вывод результатов: " << (G.nores ? C_RED "выкл" : C_GREEN "вкл")
              << C_RESET;
    std::cout << C_GRAY   << " История: " << (G.history_enabled ? "вкл" : "выкл")
              << " (переключить: /history on|off)" << C_RESET << std::endl;
    std::cout << C_GRAY   << "Подсказка: пустой Enter — отправить, '//' — отправить, "
                             "Ctrl+C во время ответа — прервать"
              << C_RESET << std::endl;
        }
    using_history();
    stifle_history(2000);
    if (G.history_enabled) read_history(READLINE_HIST_FILE.c_str());
    if (G.history_enabled) history_truncate_file(READLINE_HIST_FILE.c_str(), 2000);
    rl_attempted_completion_function = cmd_completion;

    while (true) {
        if (g_exit_requested) do_exit();
        // Сброс флага прерывания — иначе после Ctrl+C на выводе
        // следующий ввод будет сразу «прерван».
        g_stream_abort = 0;

        std::string userAnswer;
        if (!get_user_input(userAnswer)) do_exit();

        if (g_exit_requested) do_exit();

        userAnswer = expand_aliases(userAnswer);
        if (userAnswer.empty()) continue;

        // ── Специальные команды ──
        if (match_command(userAnswer, "/help")) {
            std::string ha = command_arg(userAnswer, "/help");
            print_help(ha == "all" || ha == "full" || ha == "a");
            continue;
        }
        if (userAnswer == "/save") {
            if (!G.history_enabled) {
                std::cout << C_YELLOW << "[История отключена. Включите: /history on]" << C_RESET << std::endl;
            } else {
                save_history();
            }
            continue;
        }
        if (userAnswer == "/load") {
            if (!G.history_enabled) {
                std::cout << C_YELLOW << "[История отключена. Включите: /history on]" << C_RESET << std::endl;
            } else {
                load_history();
            }
            continue;
        }
        if (match_command(userAnswer, "/history")) {
            std::string arg = command_arg(userAnswer, "/history");
            if (arg == "on") {
                G.history_enabled = true;
                save_config();
                std::cout << C_YELLOW << "[История: ВКЛЮЧЕНА]" << C_RESET << std::endl;
            } else if (arg == "off") {
                G.history_enabled = false;
                save_config();
                std::cout << C_YELLOW << "[История: ВЫКЛЮЧЕНА]" << C_RESET << std::endl;
            } else {
                print_history();
            }
            continue;
        }
        if (userAnswer == "/tokens")  { print_tokens();  continue; }
        if (match_command(userAnswer, "/cost")) {
            std::string ca = command_arg(userAnswer, "/cost");
            print_cost(ca == "live" || ca == "refresh" || ca == "api");
            continue;
        }
        if (match_command(userAnswer, "/apibase")) {
            std::string a = command_arg(userAnswer, "/apibase");
            while (!a.empty() && a[0] == ' ') a.erase(0, 1);
            if (a.empty()) {
                std::cout << C_YELLOW << "[api_base: " << normalize_api_base(G.api_base)
                          << "]" << C_RESET << std::endl;
            } else {
                G.api_base = normalize_api_base(a);
                save_config();
                std::cout << C_GREEN << "[api_base: " << G.api_base << "]" << C_RESET << std::endl;
            }
            continue;
        }
        if (match_command(userAnswer, "/FIRE")) {
            std::string fa = command_arg(userAnswer, "/FIRE");
            while (!fa.empty() && fa[0] == ' ') fa.erase(0, 1);
            while (!fa.empty() && fa.back() == ' ') fa.pop_back();
            if (fa.empty()) {
                std::cout << C_CYAN << "[FIRE: " << (G.fire ? "ВКЛ 🔥" : "ВЫКЛ")
                          << " | autorun: " << (G.autorun ? "ВКЛ" : "выкл")
                          << "]" << C_RESET << std::endl;
            } else if (fa == "on") {
                // P0.1: FIRE — режим без подтверждения bash. Требуем осознанное "yes".
                std::cout << C_RED << C_BOLD << "🔥 FIRE: ВКЛЮЧЕНИЕ ВЫПОЛНЯЕТСЯ БЕЗ ПОДТВЕРЖДЕНИЙ." << C_RESET << std::endl;
                std::cout << C_RED
                          << "  Все ```bash-блоки модели будут выполняться молча:\n"
                          << "  - без показа кода команды;\n"
                          << "  - без показа вывода;\n"
                          << "  - без запроса y/n;\n"
                          << "  - от имени текущего пользователя (доступ ко всем вашим файлам и ключам).\n"
                          << C_RESET << std::endl;
                std::cout << C_YELLOW
                          << "  Включить только если понимаете риск. Введите ровно 'yes' для входа,\n"
                          << "  что угодно другое — отмена." << C_RESET << std::endl;
                char* fire_rl = readline(C_RED "FIRE> " C_RESET);
                std::string fire_ans = fire_rl ? std::string(fire_rl) : std::string();
                if (fire_rl) free(fire_rl);
                while (!fire_ans.empty() && (fire_ans.back() == '\n' || fire_ans.back() == '\r' || fire_ans.back() == ' '))
                    fire_ans.pop_back();
                if (fire_ans != "yes") {
                    std::cout << C_GREEN << "[FIRE: отменено, режим НЕ включён]" << C_RESET << std::endl;
                    continue;
                }
                G.fire = true;
                G.autorun = true;
                G.compact_mode = true;
                save_config();
                std::cout << C_RED << C_BOLD << "🔥 FIRE: ВКЛЮЧЁН." << C_RESET << std::endl;
                std::cout << C_YELLOW << "  Выключить: /FIRE off   Аварийно: /FIRE blast"
                          << C_RESET << std::endl;
            } else if (fa == "off") {
                G.fire = false;
                G.autorun = false;
                save_config();
                std::cout << C_GREEN << "[FIRE: выключен. Autorun тоже выключен.]" << C_RESET << std::endl;
            } else if (fa == "blast") {
                G.fire = false;
                G.autorun = false;
                save_config();
                std::cout << C_RED << C_BOLD << "⛔ FIRE BLAST: аварийное отключение." << C_RESET << std::endl;
                std::cout << C_GREEN << "  FIRE и Autorun выключены." << C_RESET << std::endl;
                if (!g_last_bash_code.empty()) {
                    std::cout << C_GRAY << "  Последняя выполненная команда:" << C_RESET << std::endl;
                    std::cout << C_CODE_FG << g_last_bash_code << C_RESET << std::endl;
                } else {
                    std::cout << C_GRAY << "  (команд пока не выполнялось)" << C_RESET << std::endl;
                }
            } else {
                std::cout << C_YELLOW << "[Использование: /FIRE [on|off|blast]]" << C_RESET << std::endl;
            }
            continue;
        }
        if (match_command(userAnswer, "/time")) {
            std::string ta = command_arg(userAnswer, "/time");
            while (!ta.empty() && ta[0] == ' ') ta.erase(0, 1);
            while (!ta.empty() && ta.back() == ' ') ta.pop_back();
            if (ta == "on") {
                G.time_prefix = true;
                save_config();
                std::cout << C_GREEN << "[Time-префикс: ВКЛЮЧЁН. Дата-время МСК (дд.мм.гггг чч:мм:сс) "
                             "будет в начале каждого запроса к модели]" << C_RESET << std::endl;
                std::cout << C_GRAY  << "[Сейчас МСК: " << moscow_now_str() << "]" << C_RESET << std::endl;
            } else if (ta == "off") {
                G.time_prefix = false;
                save_config();
                std::cout << C_YELLOW << "[Time-префикс: выключен]" << C_RESET << std::endl;
            } else {
                std::cout << C_CYAN << "[Time-префикс: "
                          << (G.time_prefix ? "ВКЛ" : "выкл")
                          << " | МСК сейчас: " << moscow_now_str()
                          << " | /time on|off]" << C_RESET << std::endl;
            }
            continue;
        }
        if (userAnswer == "/nores") {
            G.nores = !G.nores;
            save_config();
            std::cout << C_YELLOW << "[Вывод результатов bash: "
                      << (G.nores ? "ВЫКЛЮЧЕН" : "включён") << "]" << C_RESET << std::endl;
            continue;
        }
        if (userAnswer == "/compact") {
            G.compact_mode = !G.compact_mode;
            save_config();
            std::cout << (G.compact_mode ? "compact on" : "compact off") << std::endl;
            continue;
        }
        if (userAnswer == "/autorun") {
            G.autorun = !G.autorun;
            save_config();
            if (G.autorun) {
                std::cout << C_RED << C_BOLD
                          << "[Autorun: ВКЛЮЧЁН ⚠ bash-блоки выполняются без подтверждения!]"
                          << C_RESET << std::endl;
                std::cout << C_YELLOW
                          << "[Опасно на системе с ключами/почтой. Выключить: /autorun]"
                          << C_RESET << std::endl;
            } else {
                std::cout << C_YELLOW << "[Autorun: выключен]" << C_RESET << std::endl;
            }
            continue;
        }
        if (userAnswer == "/dryrun") {
            g_dry_run = !g_dry_run;
            std::cout << C_YELLOW << "[Dry-run bash-блоков: "
                      << (g_dry_run ? "ВКЛЮЧЁН (команды не выполняются)" : "выключен")
                      << "]" << C_RESET << std::endl;
            continue;
        }
        if (userAnswer == "/update")  { cmd_update();  continue; }
        if (userAnswer == "/balance") { cmd_balance(); continue; }
        if (userAnswer == "/about")   { cmd_about();   continue; }


        if (match_command(userAnswer, "/new")) {
            std::string n = command_arg(userAnswer, "/new");
            if (n.empty()) n = "session_" + std::to_string(time(nullptr));
            switch_session(n); continue;
        }
        if (userAnswer == "/list") { list_sessions(); continue; }
        if (match_command(userAnswer, "/switch")) {
            std::string n = command_arg(userAnswer, "/switch");
            if (!n.empty()) switch_session(n);
            else std::cerr << C_RED << "[Укажите имя сессии]" << C_RESET << std::endl;
            continue;
        }
        if (match_command(userAnswer, "/rename")) {
            rename_session(command_arg(userAnswer, "/rename"));
            continue;
        }
        if (userAnswer == "/undo") {
            if (g_undo_stack.empty()) {
                std::cout << C_GRAY << "[Нечего отменять]" << C_RESET << std::endl;
            } else {
                G.messages = g_undo_stack.back();
                g_undo_stack.pop_back();
                std::cout << C_YELLOW << "[Отменено. Сообщений: " << G.messages.size()
                          << "]" << C_RESET << std::endl;
            }
            continue;
        }
        if (match_command(userAnswer, "/alias")) {
            std::string arg = command_arg(userAnswer, "/alias");
            size_t eq = arg.find('=');
            if (eq != std::string::npos) {
                G.aliases[arg.substr(0, eq)] = arg.substr(eq+1);
                std::cout << C_GREEN << "[Алиас сохранён]" << C_RESET << std::endl;
                save_config();
            } else if (!arg.empty()) {
                G.aliases.erase(arg);
                std::cout << C_YELLOW << "[Алиас удалён]" << C_RESET << std::endl;
                save_config();
            } else {
                for (auto& p : G.aliases) std::cout << C_CYAN << p.first << C_RESET << " = " << p.second << std::endl;
            }
            continue;
        }
        if (match_command(userAnswer, "/search")) {
            search_history(command_arg(userAnswer, "/search")); continue;
        }
        if (match_command(userAnswer, "/export")) {
            export_dialog(command_arg(userAnswer, "/export")); continue;
        }
        if (match_command(userAnswer, "/info")) {
            cmd_info(command_arg(userAnswer, "/info")); continue;
        }
        if (match_command(userAnswer, "/dump")) {
            cmd_dump(command_arg(userAnswer, "/dump")); continue;
        }
        if (userAnswer == "/exit")    { do_exit(); continue; }
        if (userAnswer == "/system")  {
            std::cout << C_MAGENTA << "[Системный промпт]:\n"
                      << G.sys_prompt << C_RESET << std::endl;
            continue;
        }
        if (match_command(userAnswer, "/voice")) {
            cmd_voice(command_arg(userAnswer, "/voice"));
            continue;
        }
        if (match_command(userAnswer, "/speak")) {
            std::string a = command_arg(userAnswer, "/speak");
            while (!a.empty() && a[0]==' ') a.erase(0,1);
            if (a.empty()) std::cout << "[Использование: /speak <текст>]" << std::endl;
            else voice_speak(a);
            continue;
        }
        if (userAnswer == "/listen") {
            if (!have_termux_api("termux-speech-to-text")) {
                std::cout << C_RED << "[voice: termux-api недоступен]" << C_RESET << std::endl;
                continue;
            }
            std::cout << C_YELLOW << "[voice: слушаю...]" << C_RESET << std::endl;
            std::string rec = voice_recognize();
            if (rec.empty()) {
                ++g_stt_empty_streak;
                if (g_stt_empty_streak >= STT_EMPTY_LIMIT && !g_stt_unavailable_notified) {
                    std::cout << C_RED << "[voice: STT-движок, похоже, недоступен на устройстве "
                              << "(" << g_stt_empty_streak << " пустых подряд). "
                              << "Проверьте: установлен ли движок распознавания речи в системе.]"
                              << C_RESET << std::endl;
                    g_stt_unavailable_notified = true;
                } else {
                    std::cout << C_GRAY << "[voice: не распознано]"
                              << (g_stt_empty_streak > 1
                                      ? " (" + std::to_string(g_stt_empty_streak) + "/"
                                            + std::to_string(STT_EMPTY_LIMIT) + ")"
                                      : std::string())
                              << C_RESET << std::endl;
                }
                continue;
            }
            g_stt_empty_streak = 0;
            std::cout << C_CYAN << "[voice]: " << rec << C_RESET << std::endl;
            if (rec[0] == '/') {
                std::cout << C_GRAY << "[voice: похоже на команду, проигнорировано]"
                          << C_RESET << std::endl;
                continue;
            }
            push_undo_snapshot();
            G.messages.push_back({{"role","user"},{"content", clamp_message_content(rec)}});
            size_t msv = G.messages.size() - 1;
            bool vab = false;
            std::string vc;
            try { vc = do_api_request(vab); } catch (...) { vc = ""; }
            process_response(vc, vab, msv);
            if (G.voice_out && !vab && !vc.empty()) {
                std::string sp = strip_for_tts(vc, G.voice_short);
                if (!sp.empty()) voice_speak(sp);
            }
            continue;
        }
        if (userAnswer == "/retry") {
            int last_assistant = -1;
            for (int i = static_cast<int>(G.messages.size()) - 1; i >= 0; --i) {
                if (G.messages[i].value("role", "") == "assistant") {
                    last_assistant = i;
                    break;
                }
            }
            int last_user = -1;
            for (int i = static_cast<int>(G.messages.size()) - 1; i >= 0; --i) {
                if (G.messages[i].value("role", "") == "user") {
                    last_user = i;
                    break;
                }
            }
            if (last_user < 0) {
                std::cout << C_GRAY << "[Нет пользовательского сообщения для повтора]"
                          << C_RESET << std::endl;
                continue;
            }
            push_undo_snapshot();
            if (last_assistant > last_user) {
                // Последний ответ — ассистент: откатываемся до user перед ним
                int user_before = -1;
                for (int i = last_assistant - 1; i >= 0; --i) {
                    if (G.messages[i].value("role", "") == "user") {
                        user_before = i;
                        break;
                    }
                }
                if (user_before >= 0) {
                    G.messages.resize(user_before + 1);
                } else {
                    G.messages.resize(last_assistant);
                }
            } else {
                // Последнее сообщение — user без ответа (например, результат
                // bash, на который модель не ответила): просто повторяем его.
                G.messages.resize(last_user + 1);
            }
            std::cout << C_YELLOW << "[Повтор запроса...]" << C_RESET << std::endl;
            // Fall through к API запросу ниже
        } else if (userAnswer == "/clear") {
            G.messages.clear();
            G.messages.push_back({{"role", "system"}, {"content", G.sys_prompt}});
            G.total_prompt_tokens     = 0;
            G.total_completion_tokens = 0;
            // Очищаем экран терминала
            std::cout << "\033[2J\033[H" << std::flush;
            std::cout << C_BOLD << C_CYAN << "=== Chat CLI ===" << C_RESET << std::endl;

    std::cout << C_YELLOW << "Модель: " << G.model << C_RESET << std::endl;
            std::cout << C_YELLOW << "[История очищена, экран очищен]" << C_RESET << std::endl;
            continue;
        } else if (match_command(userAnswer, "/delete")) {
            cmd_delete(command_arg(userAnswer, "/delete"));
            continue;
        } else if (match_command(userAnswer, "/file")) {
            cmd_file(command_arg(userAnswer, "/file"));
            continue;
        } else if (match_command(userAnswer, "/models")) {
            cmd_models(command_arg(userAnswer, "/models"));
            continue;
        } else if (match_command(userAnswer, "/model")) {
            std::string arg = command_arg(userAnswer, "/model");
            if (!arg.empty()) {
                if (AVAILABLE_MODELS.empty()) {
                    if (!load_models_cache()) refresh_models_from_api(false, true);
                    if (AVAILABLE_MODELS.empty()) AVAILABLE_MODELS = DEFAULT_MODELS;
                }
                std::string resolved, err;
                int rc = resolve_model_arg(arg, resolved, err);
                if (rc == 1) {
                    G.model = resolved;
                    save_config();
                    std::cout << C_GREEN << "[Модель: " << G.model << "]" << C_RESET << std::endl;
                } else if (rc == 2) {
                    print_models_list(LAST_MODEL_VIEW, true, 1, arg);
                    std::cout << C_YELLOW << "[" << err << ": /model N]" << C_RESET << std::endl;
                } else {
                    std::cerr << C_RED << "[" << err << "]" << C_RESET << std::endl;
                    std::cout << C_GRAY << "  подсказка: /models " << arg
                              << "  или /models refresh" << C_RESET << std::endl;
                }
            } else {
                cmd_model_select();
                save_config();
            }
            continue;
        } else if (match_command(userAnswer, "/temp")) {
            std::string arg = command_arg(userAnswer, "/temp");
            if (!arg.empty()) {
                try {
                    double t = std::stod(arg);
                    if (t >= 0.0 && t <= 2.0) {
                        G.temperature = t;
                        save_config();
                        std::cout << C_YELLOW << "[Температура: " << G.temperature
                                  << "]" << C_RESET << std::endl;
                    } else {
                        std::cerr << C_RED << "[Температура должна быть 0.0–2.0]"
                                  << C_RESET << std::endl;
                    }
                } catch (...) {
                    std::cerr << C_RED << "[Неверное значение]" << C_RESET << std::endl;
                }
            } else {
                std::cout << C_YELLOW << "[Температура: " << G.temperature
                          << "]" << C_RESET << std::endl;
            }
            continue;
        } else if (match_command(userAnswer, "/maxtokens")) {
            std::string arg = command_arg(userAnswer, "/maxtokens");
            if (!arg.empty()) {
                try {
                    int mt = std::stoi(arg);
                    if (mt > 0) {
                        G.max_tokens = mt;
                        save_config();
                        std::cout << C_YELLOW << "[max_tokens: " << G.max_tokens
                                  << "]" << C_RESET << std::endl;
                    } else {
                        std::cerr << C_RED << "[max_tokens должен быть > 0]"
                                  << C_RESET << std::endl;
                    }
                } catch (...) {
                    std::cerr << C_RED << "[Неверное значение]" << C_RESET << std::endl;
                }
            } else {
                std::cout << C_YELLOW << "[max_tokens: " << G.max_tokens
                          << "]" << C_RESET << std::endl;
            }
            continue;
        } else if (userAnswer[0] == '/') {
            std::cerr << C_RED << "[Неизвестная команда: " << userAnswer
                      << ". Введите /help]" << C_RESET << std::endl;
            continue;
        } else {
            push_undo_snapshot();
            G.messages.push_back({{"role", "user"}, {"content", clamp_message_content(userAnswer)}});
        }

        // ── API запрос ──
        size_t msgs_before = G.messages.size() - 1;
        bool aborted = false;
        std::string content;
        try {
            content = do_api_request(aborted);
        } catch (const std::exception& e) {
            std::cerr << C_RED << "[Ошибка API: " << e.what() << "]" << C_RESET << std::endl;
            content = "";
        } catch (...) {
            std::cerr << C_RED << "[Неизвестная ошибка API]" << C_RESET << std::endl;
            content = "";
        }

        if (g_exit_requested) do_exit();

        process_response(content, aborted, msgs_before);
        if (G.voice_out && !aborted && !content.empty()) {
            std::string sp = strip_for_tts(content, G.voice_short);
            if (!sp.empty()) voice_speak(sp);
        }
        // process_response already autosaves; extra silent save every N msgs
        if (G.history_enabled && G.messages.size() > 2 &&
            (G.messages.size() % HISTORY_SAVE_EVERY == 0)) {
            save_history(true);
        }
    }

    curl_global_cleanup();
    return 0;
}
