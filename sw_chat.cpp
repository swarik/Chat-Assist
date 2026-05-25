#include <iostream>
#include <unordered_set>
#include <iomanip>
#include <sys/ioctl.h>
#include <unistd.h>
#include <fstream>
#include <cstdio>
#include <cctype>
#include <unordered_map>
#include <sys/stat.h>
#include <dirent.h>
#include <algorithm>
#include <string>
#include <vector>
#include <signal.h>
#include <sstream>
#include <atomic>
#include <thread>
#include <chrono>
#include <mutex>
#include <readline/readline.h>
#include <readline/history.h>
#include <curl/curl.h>
#include "nlohmann/json.hpp"

using json = nlohmann::json;
// ─────────────────────────── Версия ───────────────────────────
#define APP_VERSION "1.0.13"

// Emoji codepoints где glibc wcwidth() неправильно возвращает 1
static std::unordered_set<int> KNOWN_WIDE_EMOJI = {
    0x1F300, 0x1F301, 0x1F302, 0x1F303, 0x1F304, 0x1F305, 0x1F306, 0x1F307,
    0x1F308, 0x1F309, 0x1F30A, 0x1F30B, 0x1F30C, 0x1F30D, 0x1F30E, 0x1F30F,
    0x1F310, 0x1F311, 0x1F312, 0x1F313, 0x1F314, 0x1F315, 0x1F316, 0x1F317,
    0x1F318, 0x1F319, 0x1F31A, 0x1F31B, 0x1F31C, 0x1F31D, 0x1F31E, 0x1F31F,
    0x1F320, 0x1F321, 0x1F322, 0x1F323, 0x1F324, 0x1F325, 0x1F326, 0x1F327,
    0x1F328, 0x1F329, 0x1F32A, 0x1F32B, 0x1F32C,
    0x1F4A0, 0x1F4A1, 0x1F4A2, 0x1F4A3, 0x1F4A4, 0x1F4A5, 0x1F4A6, 0x1F4A7,
    0x1F4A8, 0x1F4A9, 0x1F4AA, 0x1F4AB, 0x1F4AC, 0x1F4AD, 0x1F4AE, 0x1F4AF,
    0x1F525, 0x1F31F, 0x1F389, 0x1F38A, 0x1F3C6,
    0x1F3CB, 0x1F3CC, 0x1F3CD, 0x1F3CE, 0x1F3CF, 0x1F3D0, 0x1F3D1, 0x1F3D2,
    0x1F3D3, 0x1F3D4, 0x1F3D5, 0x1F3D6, 0x1F3D7, 0x1F3D8, 0x1F3D9, 0x1F3DA,
    0x1F3DB, 0x1F3DC, 0x1F3DD, 0x1F3DE, 0x1F3DF, 0x1F3E0, 0x1F3E1, 0x1F3E2,
    0x1F3E3, 0x1F3E4, 0x1F3E5, 0x1F3E6, 0x1F3E7, 0x1F3E8, 0x1F3E9, 0x1F3EA,
    0x1F3EB, 0x1F3EC, 0x1F3ED, 0x1F3EE, 0x1F3EF, 0x1F3F0,
    0x1F400, 0x1F401, 0x1F402, 0x1F403, 0x1F404, 0x1F405, 0x1F406, 0x1F407,
    0x1F408, 0x1F409, 0x1F40A, 0x1F40B, 0x1F40C, 0x1F40D, 0x1F40E, 0x1F40F,
    0x1F410, 0x1F411, 0x1F412, 0x1F413, 0x1F414, 0x1F415, 0x1F416, 0x1F417,
    0x1F418, 0x1F419, 0x1F41A, 0x1F41B, 0x1F41C, 0x1F41D, 0x1F41E, 0x1F41F,
    0x1F420, 0x1F421, 0x1F422, 0x1F423, 0x1F424, 0x1F425, 0x1F426, 0x1F427,
    0x1F428, 0x1F429, 0x1F42A, 0x1F42B, 0x1F42C, 0x1F42D, 0x1F42E, 0x1F42F,
    0x1F430, 0x1F431, 0x1F432, 0x1F433, 0x1F434, 0x1F435, 0x1F436, 0x1F437,
    0x1F438, 0x1F439, 0x1F43A, 0x1F43B, 0x1F43C, 0x1F43D, 0x1F43E, 0x1F43F,
    0x1F440, 0x1F441, 0x1F442, 0x1F443, 0x1F444, 0x1F445, 0x1F446, 0x1F447,
    0x1F448, 0x1F449, 0x1F44A, 0x1F44B, 0x1F44C, 0x1F44D, 0x1F44E, 0x1F44F,
    0x1F450, 0x1F451, 0x1F452, 0x1F453, 0x1F454, 0x1F455, 0x1F456, 0x1F457,
    0x1F458, 0x1F459, 0x1F45A, 0x1F45B, 0x1F45C, 0x1F45D, 0x1F45E, 0x1F45F,
    0x1F460, 0x1F461, 0x1F462, 0x1F463, 0x1F464, 0x1F465, 0x1F466, 0x1F467,
    0x1F468, 0x1F469, 0x1F46A, 0x1F46B, 0x1F46C, 0x1F46D, 0x1F46E, 0x1F46F,
    0x1F470, 0x1F471, 0x1F472, 0x1F473, 0x1F474, 0x1F475, 0x1F476, 0x1F477,
    0x1F478, 0x1F479, 0x1F47A, 0x1F47B, 0x1F47C, 0x1F47D, 0x1F47E, 0x1F47F,
    0x1F480, 0x1F481, 0x1F482, 0x1F483, 0x1F484, 0x1F485, 0x1F486, 0x1F487,
    0x1F488, 0x1F489, 0x1F48A, 0x1F48B, 0x1F48C, 0x1F48D, 0x1F48E, 0x1F48F,
    0x1F490, 0x1F491, 0x1F492, 0x1F493, 0x1F494, 0x1F495, 0x1F496, 0x1F497,
    0x1F498, 0x1F499, 0x1F49A, 0x1F49B, 0x1F49C, 0x1F49D, 0x1F49E, 0x1F49F,
};

// Emoji_Presentation: всегда отображается как emoji (ширина 2)
struct CpRange { uint32_t lo, hi; };
static bool cp_in_ranges(int cp, const CpRange* r, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (cp < (int)r[i].lo) return false;
        if (cp <= (int)r[i].hi) return true;
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

static int get_char_width(wchar_t wc) {
    int cp = (int)wc;
    if (is_emoji_presentation(cp)) return 2;
    if (is_emoji_codepoint(cp)) return 1; // без VS16 — текстовый
    int w = wcwidth(wc);
    return (w > 0) ? w : 0;
}

// ─────────────────────────── Цвета ───────────────────────────
#define C_RESET   "\033[0m"
#define C_GREEN   "\033[32m"
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
        int cp = (int)wc;
        if (cp >= 0x1F1E0 && cp <= 0x1F1FF) {
            // Первая буква флага
            char letter1 = 'A' + (cp - 0x1F1E6);
            i += clen;
            if (i < s.size()) {
                wchar_t next_cp = 0;
                int next_clen = mbtowc(&next_cp, s.c_str() + i, MB_CUR_MAX);
                if (next_clen > 0 && next_cp >= 0x1F1E0 && next_cp <= 0x1F1FF) {
                    char letter2 = 'A' + (next_cp - 0x1F1E6);
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

#define MAX_MESSAGES        500
#define DEFAULT_TEMPERATURE 0.7
#define DEFAULT_MAX_TOKENS  4096

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
     std::string       model          = "deepseek/deepseek-v4-pro";
    //std::string       model          = "anthropic/claude-opus-4.6";
    //std::string       model          = "anthropic/claude-opus-4.7";
    //std::string       model          = "~google/gemini-pro-latest";
    //std::string       model          = "~anthropic/claude-sonnet-latest";
    //std::string       model          = "qwen/qwen3.6-max-preview";
    //std::string       model          = "anthropic/claude-sonnet-4.6";

    std::string       sys_prompt;
    double            temperature    = DEFAULT_TEMPERATURE;
    int               max_tokens     = DEFAULT_MAX_TOKENS;
    int               total_prompt_tokens     = 0;
    int               total_completion_tokens = 0;
    bool              autorun                 = false;
    bool              history_enabled          = false;
    bool              nores                    = false; // выкл по умолчанию
    std::string       session_name             = "default";
    std::unordered_map<std::string, std::string> aliases;
};

static ChatSession G;

static const std::vector<std::string> AVAILABLE_MODELS = {
    "anthropic/claude-opus-4.6",
    "anthropic/claude-opus-4.7",
    "minimax/minimax-m2.7",
    "openai/gpt-5.2",
    "google/gemini-3.1-pro-preview",
    "x-ai/grok-4",
    "xiaomi/mimo-v2-flash",
    "deepseek/deepseek-v4-pro",
    "anthropic/claude-sonnet-4.6",
    "xiaomi/mimo-v2-pro",
    "~google/gemini-pro-latest",
    "~anthropic/claude-sonnet-latest",
    "qwen/qwen3.6-max-preview"
};

// ─────────────────────────── Сигналы ─────────────────────────
// g_exit_requested: 1 = выход из программы
// g_stream_abort:   1 = прервать текущий стриминг (Ctrl+C во время ответа)
static volatile sig_atomic_t g_exit_requested = 0;
static volatile sig_atomic_t g_stream_abort   = 0;
static volatile sig_atomic_t g_in_streaming   = 0; // 1 пока идёт стриминг
static std::mutex g_stream_mutex;  // Мьютекс для защиты потоков

static void signal_handler(int /*sig*/) {
    // Только sig_atomic_t операции — mutex нельзя использовать в обработчике сигнала (не async-signal-safe)
    if (g_in_streaming) {
        g_stream_abort = 1;
    } else {
        g_exit_requested = 1;
    }
}

// ─────────────────────────── API ключ ────────────────────────
static std::string get_api_key() {
    const char* env = getenv("OPENROUTER_API_KEY");
    if (env && std::string(env).size() > 10) return std::string(env);
    std::string home = get_home_dir();
    std::ifstream f(home + "/.config/openrouter_key");
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
        if (i + (size_t)len > input.size()) break;
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
static bool needs_variation_selector(wchar_t cp) {
    // Символы, которые по умолчанию text, но становятся emoji с FE0F
    // Misc symbols и дингбаты
    if ((cp >= 0x2600 && cp <= 0x26FF) ||   // Miscellaneous Symbols
        (cp >= 0x2700 && cp <= 0x27BF))      // Dingbats
        return true;
    return false;
}

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

        int cp = (int)wc;

        // Пропускаем zero-width символы: ZWJ, ZWNJ, variation selectors, combining
        if (cp == 0x200D || cp == 0x200C || cp == 0xFE0F || cp == 0xFE0E ||
            (cp >= 0x200B && cp <= 0x200F) || cp == 0xFEFF ||
            (cp >= 0x20D0 && cp <= 0x20FF) ||  // combining enclosing
            (cp >= 0xFE00 && cp <= 0xFE0F) ||  // variation selectors
            (cp >= 0xE0000 && cp <= 0xE01FF) || // tags
            (cp >= 0x1F3FB && cp <= 0x1F3FF)) { // skin tone modifiers
            i += clen;
            continue;
        }

        // Флаги стран: пара Regional Indicator (U+1F1E0..U+1F1FF) = 1 глиф шириной 2
        if (cp >= 0x1F1E0 && cp <= 0x1F1FF) {
            i += clen;
            // Пропускаем второй Regional Indicator если есть
            if (i < stripped.size()) {
                wchar_t next_cp = 0;
                int next_clen = mbtowc(&next_cp, stripped.c_str() + i, MB_CUR_MAX);
                if (next_clen > 0 && next_cp >= 0x1F1E0 && next_cp <= 0x1F1FF)
                    i += next_clen;
            }
            w += 2;
            continue;
        }

        // Emoji_Presentation — всегда ширина 2
        if (is_emoji_presentation(cp)) {
            w += 2;
            i += clen;
            // Пропускаем VS16 если есть
            if (i < stripped.size()) {
                wchar_t next_cp = 0;
                int next_clen = mbtowc(&next_cp, stripped.c_str() + i, MB_CUR_MAX);
                if (next_clen > 0 && (next_cp == 0xFE0F || next_cp == 0xFE0E))
                    i += next_clen;
            }
            continue;
        }

        // Emoji только с VS16 — ширина 2, без VS16 — по wcwidth
        if (is_emoji_codepoint(cp)) {
            i += clen;
            bool has_vs16 = false;
            if (i < stripped.size()) {
                wchar_t next_cp = 0;
                int next_clen = mbtowc(&next_cp, stripped.c_str() + i, MB_CUR_MAX);
                if (next_clen > 0 && (next_cp == 0xFE0F || next_cp == 0xFE0E)) {
                    has_vs16 = true;
                    i += next_clen;
                }
            }
            if (has_vs16) {
                w += 2;
            } else {
                int char_w = wcwidth(wc);
                w += (char_w > 0) ? char_w : 1;
            }
            continue;
        }

        // Обычный символ — wcwidth
        int char_w = wcwidth(wc);
        if (char_w > 0) w += char_w;
        i += clen;
    }
    return w;
}

static void render_table_row(const std::vector<std::string> &cells, const std::vector<size_t> &col_widths, bool is_header = false) {
    std::cout << C_GRAY << "\xe2\x94\x82" << C_RESET;
    for (size_t i = 0; i < cells.size(); ++i) {
        size_t col_w = (i < col_widths.size()) ? col_widths[i] : 12;
        std::string rendered = render_inline_md(cells[i]);
        size_t vis_w = visible_width(cells[i]);
        size_t pad = (vis_w < col_w) ? (col_w - vis_w) : 0;
        if (is_header)
            std::cout << " " << C_BOLD << rendered << C_RESET << std::string(pad, ' ') << " ";
        else
            std::cout << " " << rendered << std::string(pad, ' ') << " ";
        std::cout << C_GRAY << "\xe2\x94\x82" << C_RESET;
    }
    std::cout << "\n";
}

static void render_markdown(const std::string &text) {
    std::istringstream ss(text);
    std::string line;
    bool in_code = false;
    while (std::getline(ss, line)) {
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
                std::cout << C_GRAY
                          << "\xe2\x94\x94\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
                             "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
                             "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
                          << C_RESET << "\n";
            }
            continue;
        }
        if (in_code) {
            std::cout << C_GRAY << "\xe2\x94\x82 " << C_WHITE << line << C_RESET << "\n";
            continue;
        }

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
            std::cout << C_QUOTE << "  \xe2\x94\x83 " << c << C_RESET << "\n";
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
            std::cout << C_BULLET << "  \xe2\x80\xa2 " << C_RESET
                      << render_inline_md(line.substr(2)) << "\n";
            continue;
        }

        // Нумерованный список
        { size_t p=0;
          while(p<line.size()&&line[p]>='0'&&line[p]<='9') ++p;
          if (p>0&&p<line.size()&&line[p]=='.'&&p+1<line.size()&&line[p+1]==' ') {
              std::cout << C_BULLET << "  " << line.substr(0,p) << ". " << C_RESET
                        << render_inline_md(line.substr(p+2)) << "\n";
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
                std::cout << C_BULLET << "  " << ind << "\xe2\x97\xa6 " << C_RESET
                          << render_inline_md(line.substr(sp+2)) << "\n";
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
                        size_t w = visible_width(table_rows[ri][ci]);
                        if (w > table_col_widths[ci]) table_col_widths[ci] = w;
                    }
                }
                for (auto &w : table_col_widths) if (w < 3) w = 3;
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
        
        std::cout << render_inline_md(line) << "\n";
    }
    if (in_code) {
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
        G.messages.clear();
        if (!j.is_array()) return false;
        for (auto &m : j) {
            if (m.is_object()) G.messages.push_back(m);
        }
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
    std::cout << C_YELLOW << "[Системный промпт загружен из "
              << SYSTEM_PROMPT_FILE << "]" << C_RESET << std::endl;
    return content;
}

// ─────────────────────────── Обрезка контекста ───────────────
void trim_messages_if_needed() {
    if ((int)G.messages.size() <= MAX_MESSAGES) return;

    std::vector<json> trimmed;
    int sys_idx = -1;
    for (int i = 0; i < (int)G.messages.size(); ++i) {
        if (G.messages[i]["role"] == "system") { sys_idx = i; break; }
    }
    if (sys_idx >= 0) trimmed.push_back(G.messages[sys_idx]);

    int keep_count = MAX_MESSAGES - (sys_idx >= 0 ? 1 : 0);
    int start_from = (int)G.messages.size() - keep_count;
    if (start_from < 0) start_from = 0;

    for (int i = start_from; i < (int)G.messages.size(); ++i) {
        if (i == sys_idx) continue;
        trimmed.push_back(G.messages[i]);
    }
    G.messages = trimmed;
    std::cout << C_GRAY << "[Контекст обрезан до " << G.messages.size()
              << " сообщений]" << C_RESET << std::endl;
}

// ─────────────────────────── Shell exec ──────────────────────
static std::string shell_escape(const std::string& s) {
    std::string result = "'";
    for (char c : s) {
        if (c == '\'') result += "'\\''";
        else result += c;
    }
    result += "'";
    return result;
}

std::string exec_with_timeout(const std::string& cmd, int timeout_sec) {
    std::string safe_cmd = "timeout " + std::to_string(timeout_sec) +
                           " bash -c " + shell_escape(cmd) + " 2>&1";
    std::string result;
    char buffer[256];
    FILE* pipe = popen(safe_cmd.c_str(), "r");
    if (!pipe) return "[popen failed]";
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
        result += buffer;
    int ret = pclose(pipe);
    if (WIFEXITED(ret) && WEXITSTATUS(ret) == 124)
        result += "\n[ТАЙМАУТ: команда прервана после " + std::to_string(timeout_sec) + " сек]";
    if (result.size() > (size_t)MAX_CMD_OUTPUT) {
        size_t cut = MAX_CMD_OUTPUT;
        while (cut > 0 && (result[cut] & 0xC0) == 0x80) --cut;
        result = result.substr(0, cut) +
                 "\n[...вывод обрезан, превышен лимит " +
                 std::to_string(MAX_CMD_OUTPUT) + " байт...]";
    }
    return result;
}

// ─────────────────────────── Парсинг bash-команды ────────────
// Парсит все ```bash блоки из ответа
std::vector<std::string> parse_bash_blocks(const std::string &content) {
    std::vector<std::string> blocks;
    const std::string open_tag = "```bash";
    std::string::size_type pos = 0;
    while (pos < content.size()) {
        auto cnt_start = content.find(open_tag, pos);
        if (cnt_start == std::string::npos) break;
        auto search_from = cnt_start + open_tag.size();
        std::string::size_type cnt_end = std::string::npos;
        while (search_from < content.size()) {
            auto p = content.find("```", search_from);
            if (p == std::string::npos) break;
            auto after = p + 3;
            if (after >= content.size() || content[after] == '\n' ||
                content[after] == '\r'  || content[after] == ' ' ||
                !std::isalnum((unsigned char)content[after])) {
                cnt_end = p; break;
            }
            search_from = p + 3;
        }
        if (cnt_end == std::string::npos) break;
        blocks.push_back(content.substr(cnt_start + open_tag.size(),
                                        cnt_end - (cnt_start + open_tag.size())));
        pos = cnt_end + 3;
    }
    return blocks;
}

// Выполняет один bash-блок с подтверждением
// local_autorun — локальный флаг "запустить все блоки текущего пакета" (не трогает G.autorun)
std::string execute_single_bash(const std::string &bash_code, int idx, int total, bool &local_autorun) {
    if (total > 1)
        std::cout << C_YELLOW << "[Bash блок " << (idx+1) << "/" << total << "]" << C_RESET << std::endl;
    if (!G.autorun && !local_autorun) {
        char *rl = readline(C_YELLOW "[Выполнить команду? (y/n/a-все|д/н/в)]: " C_RESET);
        if (!rl) return "[Пользователь отказался выполнять эту команду]";
        std::string ans(rl); free(rl);
        if (ans == "a" || ans == "A" || ans == "в" || ans == "В") {
            local_autorun = true;  // только для текущего пакета блоков
        } else if (ans != "y" && ans != "Y" && ans != "д" && ans != "Д") {
            std::cout << C_RED << "[Блок " << (idx+1) << " пропущен]" << C_RESET << std::endl;
            return "[Пользователь отказался выполнять эту команду]";
        }
    } else if (total > 1) {
        std::cout << C_YELLOW << "[Autorun: выполняю блок " << (idx+1) << "]" << C_RESET << std::endl;
    } else {
        std::cout << C_YELLOW << "[Autorun: выполняю автоматически]" << C_RESET << std::endl;
    }
    std::cout << C_YELLOW << "[Выполняю...]" << C_RESET << std::endl;
    std::string result = exec_with_timeout(bash_code, CMD_TIMEOUT);
    if (!G.nores)
        std::cout << C_BLUE << "[Результат]:\n" << result << C_RESET << std::endl;
    return result;
}

// Обёртка совместимости
std::string execute_bash_blocks(const std::string &content) {
    auto blocks = parse_bash_blocks(content);
    if (blocks.empty()) return "";
    std::string combined;
    int total = (int)blocks.size();
    bool local_autorun = false;
    for (int idx = 0; idx < total; ++idx) {
        std::string result = execute_single_bash(blocks[idx], idx, total, local_autorun);
        if (!result.empty()) {
            if (!combined.empty()) combined += "\n---\n";
            if (total > 1) combined += "[Блок " + std::to_string(idx+1) + "]:\n";
            combined += result;
        }
    }
    return combined;
}

// ─────────────────────────── Спиннер ─────────────────────────
static std::atomic<bool> g_spinner_active{false};
static std::atomic<bool> g_spinner_stop{false}; // атомарный флаг остановки спиннера (БАГ 9)

static void spinner_thread_func(const std::string &model) {
    const char* frames[] = {"⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"};
    int idx = 0;
    while (g_spinner_active.load() && !g_spinner_stop.load()) {
        std::cout << "\r" << C_YELLOW << frames[idx % 10]
                  << " Думаю... [" << model << "]  "
                  << C_RESET << std::flush;
        idx++;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    // Стираем строку спиннера
    std::cout << "\r\033[2K" << std::flush;
}

// ─────────────────────────── API запрос ──────────────────────
// Возвращает full_content или "" при ошибке/прерывании
// aborted — true если пользователь прервал Ctrl+C
// Простой callback для накопления ответа
static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    if (g_stream_abort) return 0;
    std::string *buf = static_cast<std::string*>(userp);
    buf->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// ──────────────────── Директории и конфиг ────────────────────
static void ensure_dir(const std::string& path) { mkdir(path.c_str(), 0755); }
static void init_paths() {
    std::string home = get_home_dir();
    CONFIG_DIR     = home + "/.config/sw_chat";
    CONFIG_FILE    = CONFIG_DIR + "/config.json";
    SESSIONS_DIR   = CONFIG_DIR + "/sessions";
    HISTORY_FILE   = SESSIONS_DIR + "/" + G.session_name + ".json";
    READLINE_HIST_FILE = CONFIG_DIR + "/.readline_history";
    ensure_dir(CONFIG_DIR); ensure_dir(SESSIONS_DIR);
    G.history_file = HISTORY_FILE;
}
static void save_config() {
    try {
        json j; j["model"]=G.model; j["temperature"]=G.temperature; j["max_tokens"]=G.max_tokens;
        j["autorun"]=G.autorun; j["history_enabled"]=G.history_enabled; j["nores"]=G.nores;
    j["aliases"]=G.aliases;
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
        if(j.count("aliases")) G.aliases=j["aliases"].get<std::unordered_map<std::string,std::string>>();
    } catch(...){}
}
static void switch_session(const std::string& name) {
    if(G.history_enabled) save_history(true);
    G.session_name = name; HISTORY_FILE = SESSIONS_DIR + "/" + name + ".json";
    G.history_file = HISTORY_FILE; G.messages.clear();
    G.messages.push_back({{"role","system"},{"content",G.sys_prompt}});
    load_history(); std::cout << C_GREEN << "[Сессия: " << name << "]" << C_RESET << std::endl;
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
    std::string q=query; std::transform(q.begin(),q.end(),q.begin(),::tolower);
    bool found=false;
    for(size_t i=0;i<G.messages.size();++i) {
        std::string role=G.messages[i]["role"]; std::string cont=G.messages[i]["content"];
        std::string low=cont; std::transform(low.begin(),low.end(),low.begin(),::tolower);
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
    std::string fmt="md", file="dialog_export.md";
    size_t sp=arg.find(' '); if(sp!=std::string::npos){fmt=arg.substr(0,sp);file=arg.substr(sp+1);}
    else if(!arg.empty()){fmt=arg;file="dialog_export."+fmt;}
    std::ofstream f(file); if(!f.is_open()){std::cerr<<C_RED<<"[Не удалось создать файл]"<<C_RESET<<std::endl;return;}
    for(auto& m:G.messages){
        std::string role=m["role"]; std::string cont=m["content"];
        if(fmt=="json"){f<<m.dump(2)<<",\n";continue;}
        std::string txt=(fmt=="txt")?strip_ansi(cont):cont;
        f<<"## "<<role<<"\n"<<txt<<"\n\n";
    } std::cout << C_GREEN << "[Экспортировано в " << file << "]" << C_RESET << std::endl;
}
static void smart_trim_context() {
    const size_t MAX_CHARS = (G.max_tokens > 0 ? G.max_tokens : 4096) * 3;
    size_t total = 0;
    for(auto& m : G.messages) {
        if (m.count("content") && m["content"].is_string())
            total += m["content"].get<std::string>().size();
    }
    if(total <= MAX_CHARS) return;

    int sys_idx = -1;
    for(int i=0; i<(int)G.messages.size(); ++i) {
        if(G.messages[i].count("role") && G.messages[i]["role"] == "system") {
            sys_idx = i; break;
        }
    }

    size_t sys_size = 0;
    if (sys_idx >= 0 && G.messages[sys_idx].count("content") && G.messages[sys_idx]["content"].is_string()) {
        sys_size = G.messages[sys_idx]["content"].get<std::string>().size();
    }
    
    size_t budget = (MAX_CHARS > sys_size) ? (MAX_CHARS - sys_size) : 0;
    size_t tail_size = 0;
    int split_idx = (int)G.messages.size();

    for (int i = (int)G.messages.size() - 1; i >= 0; --i) {
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

    int erase_start = (sys_idx >= 0) ? (sys_idx + 1) : 0;
    if (split_idx > erase_start) {
        std::cout << C_GRAY << "[Обрезка контекста: удаление сообщений " << erase_start << "-" << split_idx-1 << "]" << C_RESET << std::endl;
        G.messages.erase(G.messages.begin() + erase_start, G.messages.begin() + split_idx);
    }
    std::cout << C_GRAY << "[Контекст оптимизирован: " << G.messages.size() << "]" << C_RESET << std::endl;
}
static char** cmd_completion(const char* text, int start, int end) {
    rl_attempted_completion_over = 1;
    std::vector<std::string> matches;
    std::string t(text);
    static const std::vector<std::string> cmds = {"/help","/save","/load","/clear","/history","/delete","/retry","/tokens","/model","/temp","/maxtokens","/system","/file","/autorun","/nores","/cost","/balance","/update","/about","/exit","/new","/list","/switch","/alias","/search","/export"};
    
    try {
        if (start == 0) {
            for (const auto& c : cmds) if (c.rfind(t, 0) == 0) matches.push_back(c);
        } else if (rl_line_buffer && rl_line_buffer[0] == '/' && std::string(rl_line_buffer).rfind("/model ", 0) == 0) {
            // Проверка доступности моделей
            if (!AVAILABLE_MODELS.empty()) {
                for (const auto& m : AVAILABLE_MODELS) if (m.rfind(t, 0) == 0) matches.push_back(m);
            }
        }
    } catch (...) {
        return nullptr;
    }

    if (matches.empty()) return nullptr;
    
    char** res = (char**)malloc(sizeof(char*) * (matches.size() + 1));
    if (!res) return nullptr;
    for (size_t i = 0; i < matches.size(); ++i) res[i] = strdup(matches[i].c_str());
    res[matches.size()] = nullptr;
    return res;
}
static void update_prompt() {
    size_t chars=0; for(auto& m:G.messages) { if(m.count("content") && m["content"].is_string()) chars+=m["content"].get<std::string>().size(); }
    size_t limit=G.max_tokens*3; int pct=std::min(100,(int)((chars*100)/(limit>0?limit:1)));
    std::string bar="["; for(int i=0;i<10;++i) bar+=(i<pct/10)?"#":".";
    bar+=std::to_string(pct)+"%] ";
    // Статический буфер для безопасности readline (избегаем dangling pointer)
    static char pbuf[256];
    snprintf(pbuf, sizeof(pbuf), "\001\033[1m\033[32m\002%s> \001\033[0m\002", bar.c_str());
    rl_set_prompt(pbuf);
}

std::string do_api_request(bool &aborted) {
    aborted = false;
    std::string api_key = get_api_key();
    if (api_key.empty()) return "";
    CURL *curl = curl_easy_init();
    if (!curl) return "";
    smart_trim_context();

    json jData = {
        {"model", G.model}, {"messages", G.messages},
        {"temperature", G.temperature}, {"max_tokens", G.max_tokens},
    };
    std::string jsonData = jData.dump(-1, ' ', false, json::error_handler_t::replace);
    struct curl_slist *headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    std::string auth = "Authorization: Bearer " + api_key;
    headers = curl_slist_append(headers, auth.c_str());

    struct StreamState { std::string full_content; bool header_cleared; };
    StreamState state = {"", false};

    auto write_cb = [](void *contents, size_t size, size_t nmemb, void *userp) -> size_t {
        if (g_stream_abort) return 0;
        size_t total = size * nmemb;
        StreamState* st = static_cast<StreamState*>(userp);
        if (!st->header_cleared) {
            st->header_cleared = true;
            std::cerr << "\r\033[K" << std::flush; 
        }
        st->full_content.append((char*)contents, total);
        return total;
    };

    { std::lock_guard<std::mutex> lock(g_stream_mutex); g_stream_abort = 0; g_in_streaming = 1; }
    g_spinner_stop.store(false);
    g_spinner_active.store(false); // Spinner disabled
    // Disabled spinner thread

    int retries = 3; long backoff = 2;
    CURLcode res = CURLE_OK; long http_code = 0;
    while (retries-- > 0) {
        state.full_content.clear();
        curl_easy_setopt(curl, CURLOPT_URL, "https://openrouter.ai/api/v1/chat/completions");
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonData.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)jsonData.size());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, static_cast<size_t(*)(void*,size_t,size_t,void*)>(write_cb));
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &state);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 420L);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 15L);
        curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1L);
        std::cerr << "\r\033[K" << C_YELLOW << "Думаю... [" << G.model << "] " << C_RESET << std::flush;
        res = curl_easy_perform(curl);
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        bool retryable = (res == CURLE_OPERATION_TIMEDOUT || res == CURLE_COULDNT_CONNECT || (http_code >= 500 && http_code < 600));
        if (!retryable || retries == 0) break;
        std::cout << "\r\033[2K" << C_YELLOW << "[Ошибка сети, повтор через " << backoff << "с...]" << C_RESET << std::flush;
        std::this_thread::sleep_for(std::chrono::seconds(backoff));
        backoff *= 2;
    }
    g_spinner_stop.store(true); g_spinner_active.store(false);
    // Disabled join
    { std::lock_guard<std::mutex> lock(g_stream_mutex); g_in_streaming = 0; }

    bool was_aborted = false;
    { std::lock_guard<std::mutex> lock(g_stream_mutex); if (g_stream_abort) { was_aborted = true; g_stream_abort = 0; } }
    if (was_aborted) { aborted = true; std::cout << "\n" << C_YELLOW << "[Запрос прерван]" << C_RESET << std::endl; curl_slist_free_all(headers); curl_easy_cleanup(curl); return ""; }
    if (res != CURLE_OK) { std::cerr << C_RED << "curl: " << curl_easy_strerror(res) << C_RESET << std::endl; curl_slist_free_all(headers); curl_easy_cleanup(curl); return ""; }
    if (http_code != 200) { std::cerr << C_RED << "[HTTP " << http_code << "] " << state.full_content.substr(0, 300) << C_RESET << std::endl; curl_slist_free_all(headers); curl_easy_cleanup(curl); return ""; }

    // Streaming removed: always parse full JSON
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
    
    // При стриминге текст уже выведен в консоль (сырой).
    // Повторный рендеринг создаст дубликат. Только сохраняем и ищем bash.
    if (content.empty()) return;

    if (aborted) {
        std::cout << "\n" << C_BOLD << C_CYAN << "[Ассистент]:" << C_RESET << "\n";
        render_markdown(content);
        std::cout << std::endl;
        char *rl_ans = readline(C_YELLOW "[Ответ прерван. Сохранить в историю? (y/n)]: " C_RESET);
        std::string ans;
        if (rl_ans) { ans = std::string(rl_ans); free(rl_ans); }
        if (ans != "y" && ans != "Y" && ans != "д" && ans != "Д") {
            if (msgs_before > 0 && msgs_before <= G.messages.size()) {
                G.messages.resize(msgs_before);
            } else if (!G.messages.empty() && G.messages.back()["role"] == "user") {
                G.messages.pop_back();
            }
            std::cout << C_GRAY << "[Частичный ответ отброшен]" << C_RESET << std::endl;
            return;
        }
        G.messages.push_back({{"role", "assistant"}, {"content", content}});
        return;
    }

    // ── Ищем bash-блоки в ответе ──
    const std::string open_tag = "```bash";

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

    struct BBlock { size_t tag_s, code_s, code_e, blk_e; std::string code; };
    auto find_bash_blocks = [&](const std::string &text) {
        std::vector<BBlock> bbs;
        size_t pos = 0;
        while (pos < text.size()) {
            auto ts = text.find(open_tag, pos);
            if (ts == std::string::npos) break;
            auto cs = ts + open_tag.size();
            auto ce = find_closing(text, cs);
            if (ce == std::string::npos) break;
            auto be = ce + 3;
            if (be < text.size() && text[be] == '\n') be++;
            bbs.push_back({ts, cs, ce, be, text.substr(cs, ce - cs)});
            pos = be;
        }
        return bbs;
    };

    // ── Функция: вывести ответ по частям, останавливаясь на bash-блоках ──
    // leftover — текст после последнего bash-блока (не рендерится, ждёт результатов)
    auto render_and_execute = [&](const std::string &text, std::string &leftover) -> std::string {
        leftover = "";
        // Trim trailing whitespace/newlines for rendering and bash detection
        std::string t = text;
        while (!t.empty() && (t.back() == '\n' || t.back() == '\r' || t.back() == ' ')) t.pop_back();
        
        auto bbs = find_bash_blocks(t);
        if (bbs.empty()) {
std::cout << "\n" << C_BOLD << C_CYAN << "[Ассистент]:" << C_RESET << "\n";

            render_markdown(t); // Используем обрезанный текст (t), чтобы не было лишних пустых строк

            std::cout << std::endl; // Одна пустая строка перед вводом
        return "";

        }

        std::cout << "\n" << C_BOLD << C_CYAN << "[Ассистент]:" << C_RESET << "\n";

        std::string combined_result;
        size_t cur = 0;
        int total = (int)bbs.size();
        bool local_autorun = false;

        for (int i = 0; i < total; ++i) {
            // Текст до bash-блока
            if (bbs[i].tag_s > cur) {
                render_markdown(text.substr(cur, bbs[i].tag_s - cur));
            }
            // Сам bash-блок (визуально)
            render_markdown(text.substr(bbs[i].tag_s, bbs[i].blk_e - bbs[i].tag_s));
            std::cout << std::flush;

            // Выполняем
            std::string res = execute_single_bash(bbs[i].code, i, total, local_autorun);
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
        std::cout << std::endl;
        return combined_result;
    };

    // ── Основная логика ──
    std::string leftover;
    std::string cmd_result = render_and_execute(content, leftover);

    // В контекст сохраняем весь ответ целиком (leftover будет показан после результатов)
    G.messages.push_back({{"role", "assistant"}, {"content", content}});

    // Цикл: если были bash-результаты, отправляем модели
    const int MAX_BASH_CHAIN = 7;
    for (int chain = 0; chain < MAX_BASH_CHAIN && !cmd_result.empty(); ++chain) {
        // Если был leftover — добавляем его перед результатами в сообщение user
        std::string user_msg = "[Результат выполнения команды]:\n" + cmd_result;
        G.messages.push_back({{"role", "user"}, {"content", user_msg}});

        bool chain_aborted = false;
        std::string next = do_api_request(chain_aborted);
        if (next.empty()) break;

        if (chain_aborted) {
            std::cout << "\n" << C_BOLD << C_CYAN << "[Ассистент]:" << C_RESET << "\n";
            render_markdown(next);
            std::cout << "\n" << C_YELLOW << "[Ответ прерван]" << C_RESET << std::endl;
            break;
        }

        std::string next_leftover;
        cmd_result = render_and_execute(next, next_leftover);
        G.messages.push_back({{"role", "assistant"}, {"content", next}});
        leftover = next_leftover;
    }

    // Если остался leftover и bash-цикл завершился (cmd_result пуст) — рендерим его
    if (!leftover.empty() && cmd_result.empty()) {
        render_markdown(leftover);
        std::cout << std::endl;
    }

    // Автосохранение: пишем всегда, уведомляем раз в 24 сообщения
    if (G.history_enabled && G.messages.size() > 2) save_history(G.messages.size() % 24 != 0);
}

// ─────────────────────────── Команды ──────────────────────────
void cmd_update() {
    std::string home = get_home_dir();
    std::string url = "https://raw.githubusercontent.com/swarik/Chat-Assist/main/sw_chat.cpp";
    std::string new_src = home + "/tmp/sw_chat_new.cpp";
    std::string new_bin = home + "/tmp/sw_chat_new";
    std::string cur_bin = home + "/sw_chat";

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
            while (std::getline(ss, token, '.')) parts.push_back(std::stoi(token));
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

    // Запрашиваем согласие пользователя
    {
        char *rl_ans = readline(C_YELLOW "[update] Установить обновление? (y/n): " C_RESET);
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
    std::string compile_cmd = "g++ -std=c++17 -O2 -I" + home + "/.local/include -o "
        + new_bin + " " + new_src + " -lreadline -lcurl -lpthread 2>&1";
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

    // 7. Заменить старые файлы (mv атомарно, не блокируется запущенным процессом)
    std::string old_bin = cur_bin + ".old";
    std::string mv_cmd = "mv " + cur_bin + " " + old_bin + " && "
        + "mv " + new_bin + " " + cur_bin + " && chmod +x " + cur_bin + " && "
        + "mv " + new_src + " " + home + "/sw_chat.cpp && "
        + "rm -f " + old_bin;
    int mv_res = system(mv_cmd.c_str());
    if (mv_res != 0) {
        std::cerr << C_RED << "[update: не удалось заменить файлы]" << C_RESET << std::endl;
        return;
    }

    std::cout << C_GREEN << C_BOLD << "[update] Обновление установлено! Перезапуск..." << C_RESET << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // 9. exec — заменить текущий процесс
    execl(cur_bin.c_str(), cur_bin.c_str(), "--restore-session", (char*)NULL);

    // Если execl не сработал
    std::cerr << C_RED << "[update: exec failed]" << C_RESET << std::endl;
}

void cmd_balance() {
    std::string api_key = get_api_key();
    if (api_key.empty()) return;

    CURL *curl = curl_easy_init();
    if (!curl) {
        std::cerr << C_RED << "[balance: curl init failed]" << C_RESET << std::endl;
        return;
    }

    std::string response_body;
    std::string auth = "Authorization: Bearer " + api_key;
    struct curl_slist *headers = nullptr;
    headers = curl_slist_append(headers, auth.c_str());

    curl_easy_setopt(curl, CURLOPT_URL, "https://openrouter.ai/api/v1/credits");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 15L);

    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        std::cerr << C_RED << "[balance: curl failed: " << curl_easy_strerror(res) << "]" << C_RESET << std::endl;
        return;
    }

    if (http_code != 200) {
        std::cerr << C_RED << "[balance: HTTP " << http_code << "]" << std::endl << response_body << C_RESET << std::endl;
        return;
    }

    try {
        json j = json::parse(response_body);
        json data = j.count("data") ? j["data"] : j;

        double total_credits = data.value("total_credits", 0.0);
        double total_usage   = data.value("total_usage", 0.0);
        double remaining     = total_credits - total_usage;

        std::cout << C_CYAN << C_BOLD << "  === OpenRouter Balance ===" << C_RESET << std::endl;
        std::cout << "  Credits:  " << C_GREEN << "$" << std::fixed << std::setprecision(4) << total_credits << C_RESET << std::endl;
        std::cout << "  Used:     " << C_YELLOW << "$" << total_usage << C_RESET << std::endl;
        std::cout << "  Remain:   " << (remaining > 1.0 ? C_GREEN : C_RED) << "$" << remaining << C_RESET << std::endl;
    } catch (const std::exception& e) {
        std::cerr << C_RED << "[balance: parse error: " << e.what() << "]" << std::endl << "Raw: " << response_body << C_RESET << std::endl;
    }
}

// ─────────────────────────── Команда /about ──────────────────
void cmd_about() {
    std::cout << C_CYAN << C_BOLD << "  === Chat CLI ===" << C_RESET << std::endl;
    std::cout << "  Версия:   " << C_GREEN << APP_VERSION << C_RESET << std::endl;
    std::cout << "  Модель:   " << C_GREEN << G.model << C_RESET << std::endl;
    std::cout << "  Temp:     " << C_GREEN << G.temperature << C_RESET << std::endl;
    std::cout << "  Max tok:  " << C_GREEN << G.max_tokens << C_RESET << std::endl;
    std::cout << "  Autorun:  " << (G.autorun ? C_GREEN "вкл" : C_RED "выкл") << C_RESET << std::endl;
    std::cout << "  History:  " << (G.history_enabled ? C_GREEN "вкл" : C_RED "выкл") << C_RESET << std::endl;
    std::cout << "  NoRes:    " << (G.nores ? C_RED "вкл" : C_GREEN "выкл") << " (скрытие вывода bash)" << C_RESET << std::endl;

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
void do_exit() {
    g_exit_requested = 1;
    
    // Останавливаем спиннер если он активен
    if (g_spinner_active.load()) {
        g_spinner_stop.store(true);
        g_spinner_active.store(false);
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
    
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

void cmd_model_select() {
    std::cout << C_YELLOW << "\n╔══════════════════════════════════════════════════╗\n";
    std::cout << "║             Доступные модели                     ║\n";
    std::cout << "╠══════════════════════════════════════════════════╣" << C_RESET << "\n";
    for (size_t i = 0; i < AVAILABLE_MODELS.size(); ++i) {
        bool is_current = (AVAILABLE_MODELS[i] == G.model);
        if (is_current)
            std::cout << C_GREEN << C_BOLD;
        else
            std::cout << C_CYAN;
        printf("  %2zu) %s", i + 1, AVAILABLE_MODELS[i].c_str());
        if (is_current) std::cout << "  ◄── текущая";
        std::cout << C_RESET << "\n";
    }
    std::cout << C_YELLOW << "╚══════════════════════════════════════════════════╝" << C_RESET << "\n";
    char *rl_choice = readline(C_YELLOW "[Номер модели или Enter для отмены]: " C_RESET);
    if (!rl_choice) return;
    std::string choice(rl_choice);
    free(rl_choice);
    if (choice.empty()) return;
    try {
        int idx = std::stoi(choice);
        if (idx >= 1 && idx <= (int)AVAILABLE_MODELS.size()) {
            G.model = AVAILABLE_MODELS[idx - 1];
            std::cout << C_GREEN << "[Модель: " << G.model << "]" << C_RESET << std::endl;
        } else {
            std::cerr << C_RED << "[Неверный номер]" << C_RESET << std::endl;
        }
    } catch (...) {
        // Может быть введено имя модели напрямую
        G.model = choice;
        std::cout << C_GREEN << "[Модель: " << G.model << "]" << C_RESET << std::endl;
    }
}

void print_help() {
    std::cout << C_YELLOW
        << "Специальные команды:\n"
        << "  /save              — сохранить историю\n"
        << "  /load              — загрузить историю\n"
        << "  /clear             — очистить историю диалога\n"
        << "  /history [on|off]  — показать историю / вкл-выкл сохранение\n"
        << "  /delete N          — удалить сообщение N из истории\n"
        << "  /retry             — повторить последний запрос\n"
        << "  /tokens            — показать использование токенов\n"
        << "  /model [name|N]    — выбор модели из списка / по имени / по номеру\n"
        << "  /temp [0.0-2.0]    — показать/сменить температуру\n"
        << "  /maxtokens [N]     — показать/сменить max_tokens\n"
        << "  /system            — показать системный промпт\n"
        << "  /file <path> [msg] — загрузить файл и задать вопрос\n"
        << "  /autorun           — вкл/выкл авто-выполнение bash\n  /nores             — вкл/выкл вывод результатов bash\n"
        << "  /cost              — стоимость токенов в $\n"
        << "  /balance           — проверить баланс OpenRouter\n"
        << "  /update            — обновление программы\n"
        << "  /about             — информация о программе\n"
        << "  /new [name]        — создать новую сессию\n"
        << "  /list              — список сессий\n"
        << "  /switch <name>     — переключить сессию\n"
        << "  /alias k=v         — создать/удалить/показать алиасы\n"
        << "  /search <text>     — поиск по истории\n"
        << "  /export [fmt] [f]  — экспорт диалога (md/txt/json)\n"
        << "  /help              — эта справка\n"
        << "  /exit              — выход\n"
        << "\nМногострочный ввод:\n"
        << "  Пустой Enter      — отправить сообщение\n"
        << "  //                 — отправить сообщение (конец ввода)\n"
        << "  .                  — вставить пустую строку\n"
        << "\nВо время получения ответа:\n"
        << "  Ctrl+C             — прервать вывод ответа\n"
        << C_RESET;
}

void print_history() {
    std::cout << C_YELLOW << "[История диалога (" << G.messages.size()
              << " сообщений)]:" << C_RESET << std::endl;
    for (size_t i = 0; i < G.messages.size(); ++i) {
        std::string role = G.messages[i]["role"];
        std::string cont = G.messages[i]["content"];
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
        if (idx < 0 || idx >= (int)G.messages.size()) {
            std::cerr << C_RED << "[Неверный индекс: " << idx << "]" << C_RESET << std::endl;
            return;
        }
        if (G.messages[idx]["role"] == "system") {
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
    {"anthropic/claude-opus-4.7",       5.0,   25.0},
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

void print_cost() {
    double p_price = 0, c_price = 0;
    bool found = false;
    for (int i = 0; KNOWN_PRICING[i].model_prefix != nullptr; ++i) {
        if (G.model.find(KNOWN_PRICING[i].model_prefix) == 0) {
            p_price = KNOWN_PRICING[i].prompt_per_mtok;
            c_price = KNOWN_PRICING[i].completion_per_mtok;
            found = true;
            break;
        }
    }
    double prompt_cost = (G.total_prompt_tokens / 1000000.0) * p_price;
    double completion_cost = (G.total_completion_tokens / 1000000.0) * c_price;
    double total_cost = prompt_cost + completion_cost;
    int total_tokens = G.total_prompt_tokens + G.total_completion_tokens;

    std::cout << C_MAGENTA << "\n  Использование токенов" << C_RESET << "\n";
    std::cout << C_GRAY << "  ────────────────────────────────" << C_RESET << "\n";
    std::cout << C_MAGENTA << "  Модель:\t" << C_RESET << G.model << "\n";
    std::cout << C_MAGENTA << "  Промпт:\t" << C_RESET << G.total_prompt_tokens << " токенов\n";
    std::cout << C_MAGENTA << "  Ответы:\t" << C_RESET << G.total_completion_tokens << " токенов\n";
    std::cout << C_MAGENTA << "  Всего:\t" << C_RESET << total_tokens << " токенов\n";
    if (found) {
        std::cout << C_GRAY << "  ────────────────────────────────" << C_RESET << "\n";
        printf("  Промпт:\t$%.4f\n", prompt_cost);
        printf("  Ответы:\t$%.4f\n", completion_cost);
        printf("  Итого:\t$%.4f\n", total_cost);
        printf("  ($%.2f/$%.2f за 1M токенов)\n", p_price, c_price);
    } else {
        std::cout << C_GRAY << "\n  Цены для модели не найдены" << C_RESET << "\n";
    }
    std::cout << std::endl;
}

// ─────────────────────────── Ввод пользователя ───────────────
static bool get_user_input(std::string &out) {
    std::string result;
    bool first_line = true;
    // bool multiline = false; // БАГ 7: переменная объявлялась но нигде не читалась
    int  line_num   = 1;

    while (true) {
        if (g_exit_requested) return false;

        std::string prompt = first_line
            ? "\n\001\033[1m\033[32m\002> \001\033[0m\002"
            : ("\001\033[32m\002" + std::to_string(line_num) + "> \001\033[0m\002");

        std::cout.flush(); fflush(stdout); // Сброс буферов перед вызовом readline
        std::cout.flush(); fflush(stdout); // Сброс буферов перед вызовом readline
        char *line = readline(prompt.c_str());

        if (!line) {
            // EOF (Ctrl+D)
            if (!result.empty()) {
                out = result;
                if (G.history_enabled) add_history(result.size() <= 500
                    ? result.c_str()
                    : (result.substr(0, 500) + "...").c_str());
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
            std::cout << C_GRAY
                      << "[Многострочный режим: пустой Enter — отправить, '.' — пустая строка, '//' — отправить]"
                      << C_RESET << std::endl;
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
            std::string hist = result.size() > 500 ? result.substr(0, 500) + "..." : result;
            add_history(hist.c_str());
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

// ─────────────────────────── main ────────────────────────────
int main(int argc, char *argv[]) {

std::setlocale(LC_ALL, "");
    // std::ios::sync_with_stdio(false); // Отключаем синхронизацию для readline
    rl_catch_signals = 0;            // Мы сами обрабатываем сигналы
    rl_catch_sigwinch = 1;           // Readline сам обрабатывает ресайз окна
    // std::ios::sync_with_stdio(false); // Отключаем синхронизацию для readline
    rl_catch_signals = 0;            // Мы сами обрабатываем сигналы
    rl_catch_sigwinch = 1;           // Readline сам обрабатывает ресайз окна
    // Инициализируем пути
    std::string home = get_home_dir();
    HISTORY_FILE       = home + "/tmp/chat_history.json";
    SYSTEM_PROMPT_FILE = home + "/tmp/system_prompt.txt";
    READLINE_HIST_FILE = home + "/tmp/.chat_readline_history";
    G.history_file     = HISTORY_FILE;

    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);

    G.sys_prompt = load_system_prompt();
    if (G.sys_prompt.empty()) {
        G.sys_prompt =
            "То, что ты выведешь после ```bash будет сразу исполняться в системе через функцию system();. "
            "Используй максимально аккуратно, чтобы не навредить системе !!! "
            "Всегда придерживайся правила: несколько bash-блоков могут быть в твоём ответе, все будут выполнены последовательно. "
            "При выводе тобой bash-блока ничего больше не выводить, пока я разрешу или не разрешу."
            "Все инструкции, что указаны здесь выше ты должен постоянно помнить и не нарушать. "
            "ЭТО ВАЖНО! Результат выполнения команды будет добавлен к твоему сообщению автоматически. "
            "В папке ~/tmp возможно будет файл memo.md это твоя память. "
            "Если необходимо сделать запись в memo.md, то сохраняй самое важное, максимум три - пять строк, ДОПИСЫВАЯ в файл.";
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
    int real_args = 0;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) != "--restore-session") real_args++;
    }
    bool has_args  = real_args > 0;

    if (pipe_mode || has_args) {
        std::string message;
        if (has_args) {
            for (int i = 1; i < argc; ++i) {
                if (i > 1) message += " ";
                message += argv[i];
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
        G.messages.push_back({{"role", "user"}, {"content", message}});
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
            std::cout << "\n";
            render_markdown(content);
            std::cout << std::endl;
        }
        curl_global_cleanup();
        return 0;
    }

    // ── Интерактивный режим ──
    init_paths();
    load_config();

    std::cout << C_BOLD << C_CYAN << "=== Chat CLI ===" << C_RESET << std::endl;
    std::cout << C_YELLOW << "Модель: " << G.model << C_RESET << std::endl;
    std::cout << C_YELLOW << "Введите /help для справки" << C_RESET << std::endl;
    std::cout << C_GRAY   << "Autorun: " << (G.autorun ? "вкл" : "выкл")
              << " (переключить: /autorun)" << C_RESET
              << C_GRAY << " Вывод результатов: " << (G.nores ? C_RED "выкл" : C_GREEN "вкл")
              << C_RESET;
    std::cout << C_GRAY   << " История: " << (G.history_enabled ? "вкл" : "выкл")
              << " (переключить: /history on|off)" << C_RESET << std::endl;
    std::cout << C_GRAY   << "Подсказка: пустой Enter — отправить, '//' — отправить, "
                             "Ctrl+C во время ответа — прервать"
              << C_RESET << std::endl;
    using_history();
    if (G.history_enabled) read_history(READLINE_HIST_FILE.c_str());
    rl_attempted_completion_function = cmd_completion;

    while (true) {
        if (g_exit_requested) do_exit();

        std::string userAnswer;
        if (!get_user_input(userAnswer)) do_exit();

        if (g_exit_requested) do_exit();

        userAnswer = expand_aliases(userAnswer);
        if (userAnswer.empty()) continue;

        // ── Специальные команды ──
        if (userAnswer == "/help")    { print_help();    continue; }
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
                std::cout << C_YELLOW << "[История: ВКЛЮЧЕНА]" << C_RESET << std::endl;
            } else if (arg == "off") {
                G.history_enabled = false;
                std::cout << C_YELLOW << "[История: ВЫКЛЮЧЕНА]" << C_RESET << std::endl;
            } else {
                print_history();
            }
            continue;
        }
        if (userAnswer == "/tokens")  { print_tokens();  continue; }
        if (userAnswer == "/cost")    { print_cost();    continue; }
        if (userAnswer == "/nores") {
            G.nores = !G.nores;
            std::cout << C_YELLOW << "[Вывод результатов bash: "
                      << (G.nores ? "ВЫКЛЮЧЕН" : "включён") << "]" << C_RESET << std::endl;
            continue;
        }
        if (userAnswer == "/autorun") {
            G.autorun = !G.autorun;
            std::cout << C_YELLOW << "[Autorun: "
                      << (G.autorun ? "ВКЛЮЧЁН ⚡" : "выключен")
                      << "]" << C_RESET << std::endl;
            continue;
        }
        if (userAnswer == "/update")  { cmd_update();  continue; }
        if (userAnswer == "/balance") { cmd_balance(); continue; }
        if (userAnswer == "/about")   { cmd_about();   continue; }


        if (match_command(userAnswer, "/new")) {
            std::string n = command_arg(userAnswer, "/new");
            if (n.empty()) n = "session_" + std::to_string(time(0));
            switch_session(n); continue;
        }
        if (userAnswer == "/list") { list_sessions(); continue; }
        if (match_command(userAnswer, "/switch")) {
            std::string n = command_arg(userAnswer, "/switch");
            if (!n.empty()) switch_session(n);
            else std::cerr << C_RED << "[Укажите имя сессии]" << C_RESET << std::endl;
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
        if (userAnswer == "/exit")    { do_exit(); }
        if (userAnswer == "/system")  {
            std::cout << C_MAGENTA << "[Системный промпт]:\n"
                      << G.sys_prompt << C_RESET << std::endl;
            continue;
        }
        if (userAnswer == "/retry") {
            int last_assistant = -1;
            for (int i = (int)G.messages.size() - 1; i >= 0; --i) {
                if (G.messages[i]["role"] == "assistant") {
                    last_assistant = i;
                    break;
                }
            }
            if (last_assistant > 0) {
                int user_before = -1;
                for (int i = last_assistant - 1; i >= 0; --i) {
                    if (G.messages[i]["role"] == "user") {
                        user_before = i;
                        break;
                    }
                }
                if (user_before >= 0) {
                    G.messages.resize(user_before + 1);
                } else {
                    G.messages.resize(last_assistant);
                }
                std::cout << C_YELLOW << "[Повтор запроса...]" << C_RESET << std::endl;
                // Fall through к API запросу ниже
            } else {
                std::cout << C_GRAY << "[Нет ответа ассистента для повтора]"
                          << C_RESET << std::endl;
                continue;
            }
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
        } else if (match_command(userAnswer, "/model")) {
            std::string arg = command_arg(userAnswer, "/model");
            if (!arg.empty()) {
                // Проверяем — может это номер из списка
                try {
                    int idx = std::stoi(arg);
                    if (idx >= 1 && idx <= (int)AVAILABLE_MODELS.size()) {
                        G.model = AVAILABLE_MODELS[idx - 1];
                    } else {
                        G.model = arg;
                    }
                } catch (...) {
                    G.model = arg;
                }
                std::cout << C_GREEN << "[Модель: " << G.model << "]" << C_RESET << std::endl;
            } else {
                cmd_model_select();
            }
            continue;
        } else if (match_command(userAnswer, "/temp")) {
            std::string arg = command_arg(userAnswer, "/temp");
            if (!arg.empty()) {
                try {
                    double t = std::stod(arg);
                    if (t >= 0.0 && t <= 2.0) {
                        G.temperature = t;
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
            G.messages.push_back({{"role", "user"}, {"content", userAnswer}});
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
        if (G.history_enabled) save_history(true);
        update_prompt();
    }

    curl_global_cleanup();
    return 0;
}
