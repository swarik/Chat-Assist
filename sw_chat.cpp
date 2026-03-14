#include <iostream>
#include <sys/ioctl.h>
#include <unistd.h>
#include <fstream>
#include <cstdio>
#include <string>
#include <vector>
#include <signal.h>
#include <unistd.h>
#include <sstream>
#include <readline/readline.h>
#include <readline/history.h>
#include <curl/curl.h>
#include "nlohmann/json.hpp"

using json = nlohmann::json;

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

// ─────────────────────────── Вспомогательная функция ─────────
static std::string get_home_dir() {
    const char* h = getenv("HOME");
    if (h) return std::string(h);
    return "/tmp";  // fallback если HOME не задан
}

// ─────────────────────────── Константы ───────────────────────
#define HISTORY_FILE        (get_home_dir() + "/tmp/chat_history.json")
#define SYSTEM_PROMPT_FILE  (get_home_dir() + "/tmp/system_prompt.txt")
#define READLINE_HIST_FILE  (get_home_dir() + "/tmp/.chat_readline_history")
#define CMD_TIMEOUT         85
#define MAX_CMD_OUTPUT      96000
#define MAX_MESSAGES        200
#define DEFAULT_TEMPERATURE 0.7
#define DEFAULT_MAX_TOKENS  12000

// ─────────────────────────── Глобальное состояние ────────────
struct ChatSession {
    std::vector<json> messages;
    std::string       history_file   = HISTORY_FILE;

 //std::string       model          = "minimax/minimax-m2.5";
 //std::string       model          = "anthropic/claude-sonnet-4";
 //std::string       model          = "openai/gpt-5.2";
 //std::string       model          = "google/gemini-3.1-pro-preview";
 //std::string       model          = "x-ai/grok-4";
 //std::string       model          = "qwen/qwen3-max-thinking";
 //std::string       model          = "xiaomi/mimo-v2-flash";
 //std::string       model          = "nex-agi/deepseek-v3.1-nex-n1";
 //std::string       model          = "anthropic/claude-opus-4.6";
 std::string       model          = "anthropic/claude-sonnet-4.6";

    std::string       sys_prompt;
    double            temperature    = DEFAULT_TEMPERATURE;
    int               max_tokens     = DEFAULT_MAX_TOKENS;
    int               total_prompt_tokens     = 0;
    int               total_completion_tokens = 0;
};

static ChatSession G;
static volatile sig_atomic_t g_exit_requested = 0;

// ─────────────────────────── API ключ ────────────────────────
static std::string get_api_key() {
    const char* env = getenv("OPENROUTER_API_KEY");
    if (env && std::string(env).size() > 10) return std::string(env);
    // Попытка читать из файла
    std::string home = get_home_dir();
    std::ifstream f(home + "/.config/openrouter_key");
    if (f.is_open()) {
        std::string key;
        std::getline(f, key);
        // Убираем пробелы/переносы в конце
        while (!key.empty() && (key.back() == '\n' || key.back() == '\r' || key.back() == ' '))
            key.pop_back();
        if (key.size() > 10) return key;
    }
    std::cerr << C_RED << "[ОШИБКА: API ключ не найден! Задайте OPENROUTER_API_KEY или создайте ~/.config/openrouter_key]" << C_RESET << std::endl;
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

// ─────────────────────────── CURL callback ───────────────────
static size_t WriteCallback(void *contents, size_t size, size_t nmemb, std::string *s) {
    s->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// ─────────────────────────── Streaming SSE ───────────────────
struct StreamContext {
    std::string buffer;
    std::string full_content;
    int         prompt_tokens     = 0;
    int         completion_tokens = 0;
    bool        first_token       = true;
    bool        done              = false;
    int         lines_printed     = 0;  // кол-во \n выведенных сырым стримингом
};

static size_t StreamCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    StreamContext *ctx = static_cast<StreamContext*>(userp);
    std::string chunk((char*)contents, size * nmemb);
    ctx->buffer += chunk;

    // Обрабатываем построчно
    std::string::size_type pos;
    while ((pos = ctx->buffer.find("\n")) != std::string::npos) {
        std::string line = ctx->buffer.substr(0, pos);
        ctx->buffer.erase(0, pos + 1);

        // Убираем \r если есть
        if (!line.empty() && line.back() == '\r') line.pop_back();

        // Пустые строки — разделители SSE
        if (line.empty()) continue;

        // Финальный маркер
        if (line == "data: [DONE]") {
            ctx->done = true;
            continue;
        }

        // Парсим "data: {...}"
        if (line.rfind("data: ", 0) != 0) continue;
        std::string json_str = line.substr(6);

        try {
            json j = json::parse(json_str);

            // Извлекаем дельту контента
            if (j.contains("choices") && !j["choices"].empty()) {
                auto &choice = j["choices"][0];
                if (choice.contains("delta") && choice["delta"].contains("content")) {
                    std::string token = choice["delta"]["content"];
                    if (!token.empty()) {
                        // Выводим токен сразу — настоящий стриминг
                        std::cout << token << std::flush;
                        for (char c : token)
                            if (c == '\n') ctx->lines_printed++;
                        ctx->full_content += token;
                    }
                }
            }

            // Извлекаем usage если есть (обычно в последнем чанке)
            if (j.contains("usage")) {
                ctx->prompt_tokens     = j["usage"].value("prompt_tokens", 0);
                ctx->completion_tokens = j["usage"].value("completion_tokens", 0);
            }
        } catch (...) {
            // Некорректный JSON — пропускаем
        }
    }

    return size * nmemb;
}


// ─────────────────────────── История ─────────────────────────
void save_history() {
    try {
        json j = json::array();
        for (auto &m : G.messages) j.push_back(m);
        std::ofstream f(G.history_file);
        if (f.is_open()) {
            f << j.dump(2, ' ', false, json::error_handler_t::replace);
            std::cout << C_YELLOW << "[История сохранена: " << G.history_file << "]" << C_RESET << std::endl;
        } else {
            std::cerr << C_RED << "[Не удалось открыть файл истории для записи]" << C_RESET << std::endl;
        }
    } catch (...) {
        std::cerr << C_RED << "[Ошибка сохранения истории]" << C_RESET << std::endl;
    }
}

bool load_history() {
    std::ifstream f(G.history_file);
    if (!f.is_open()) return false;
    try {
        json j = json::parse(f);
        G.messages.clear();
        for (auto &m : j) G.messages.push_back(m);
        // Проверяем что первое сообщение — системный промпт
        if (G.messages.empty() || G.messages[0]["role"] != "system") {
            // Вставляем системный промпт в начало
            G.messages.insert(G.messages.begin(),
                {{"role", "system"}, {"content", G.sys_prompt}});
        }
        std::cout << C_YELLOW << "[История загружена: " << G.messages.size()
                  << " сообщений]" << C_RESET << std::endl;
        return true;
    } catch (...) {
        std::cerr << C_RED << "[Ошибка загрузки истории]" << C_RESET << std::endl;
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

    // Ищем системный промпт (может быть на позиции 0)
    int sys_idx = -1;
    for (int i = 0; i < (int)G.messages.size(); ++i) {
        if (G.messages[i]["role"] == "system") {
            sys_idx = i;
            break;
        }
    }

    // Сохраняем системный промпт если есть
    if (sys_idx >= 0) {
        trimmed.push_back(G.messages[sys_idx]);
    }

    // Оставляем последние MAX_MESSAGES-1 сообщений (без system)
    int keep_count = MAX_MESSAGES - (sys_idx >= 0 ? 1 : 0);
    int start_from = (int)G.messages.size() - keep_count;
    if (start_from < 0) start_from = 0;

    for (int i = start_from; i < (int)G.messages.size(); ++i) {
        if (i == sys_idx) continue; // уже добавлен
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
    // Ограничение размера вывода (по байтам, с защитой UTF-8 на границе)
    if (result.size() > (size_t)MAX_CMD_OUTPUT) {
        size_t cut = MAX_CMD_OUTPUT;
        // Не разрезаем UTF-8 символ: откатываемся назад если попали в continuation byte
        while (cut > 0 && (result[cut] & 0xC0) == 0x80) --cut;
        result = result.substr(0, cut) +
                 "\n[...вывод обрезан, превышен лимит " + std::to_string(MAX_CMD_OUTPUT) + " байт...]";
    }
    return result;
}

// ─────────────────────────── Парсинг команды ─────────────────
std::string for_com_parse(const std::string &content) {
    const std::string open_tag = "```bash";
    std::string::size_type cnt_start = content.find(open_tag);
    if (cnt_start == std::string::npos) return "";

    std::string::size_type search_from = cnt_start + open_tag.size();
    std::string::size_type cnt_end = std::string::npos;

    while (search_from < content.size()) {
        std::string::size_type pos = content.find("```", search_from);
        if (pos == std::string::npos) break;
        std::string::size_type after = pos + 3;
        if (after >= content.size() || content[after] == '\n' ||
            content[after] == '\r'  || content[after] == ' ') {
            cnt_end = pos;
            break;
        }
        search_from = pos + 3;
    }

    if (cnt_end == std::string::npos) return "";

    std::string com_cont = content.substr(cnt_start + open_tag.size(),
                                          cnt_end - (cnt_start + open_tag.size()));

    std::cout << C_YELLOW << "[Выполнить команду? (y/n|д/н)]: " << C_RESET << std::flush;
    std::string confirm;
    if (!std::getline(std::cin, confirm)) return "";
    if (confirm != "y" && confirm != "Y" && confirm != "д" && confirm != "Д") {
        std::cout << C_RED << "[Выполнение отменено пользователем]" << C_RESET << std::endl;
        return "";
    }
    std::cout << C_YELLOW << "[Выполняю...]" << C_RESET << std::endl;
    std::string result = exec_with_timeout(com_cont, CMD_TIMEOUT);
    std::cout << C_BLUE << "[Результат]:\n" << result << C_RESET << std::endl;
    return result;
}

// ─────────────────────────── Вывод ───────────────────────────
void print_assistant(const std::string &content) {
    std::cout << C_BOLD << C_CYAN << "\n[Ассистент]:" << C_RESET << std::endl;
    std::cout << content << std::endl;
}

// --- Markdown rendering ---
#define C_ITALIC    "\033[3m"
#define C_BG_GRAY   "\033[48;5;236m"
#define C_WHITE     "\033[97m"
#define C_CODE_FG   "\033[93m"
#define C_QUOTE     "\033[36;3m"
#define C_BULLET    "\033[33m"
#define C_H1        "\033[1;35m"
#define C_H2        "\033[1;36m"
#define C_H3        "\033[1;33m"

static std::string render_inline_md(const std::string &line) {
    std::string out;
    out.reserve(line.size() * 2);
    size_t i = 0, len = line.size();
    while (i < len) {
        if (line[i] == '`' && (i+1 < len) && line[i+1] != '`') {
            size_t end = line.find('`', i+1);
            if (end != std::string::npos) {
                out += C_BG_GRAY; out += C_CODE_FG;
                out += line.substr(i+1, end-i-1);
                out += C_RESET;
                i = end + 1; continue;
            }
        }
        if (i+1 < len && line[i] == '*' && line[i+1] == '*') {
            size_t end = line.find("**", i+2);
            if (end != std::string::npos) {
                out += C_BOLD;
                out += line.substr(i+2, end-i-2);
                out += C_RESET;
                i = end + 2; continue;
            }
        }
        if (i+1 < len && line[i] == '_' && line[i+1] == '_') {
            size_t end = line.find("__", i+2);
            if (end != std::string::npos) {
                out += C_BOLD;
                out += line.substr(i+2, end-i-2);
                out += C_RESET;
                i = end + 2; continue;
            }
        }
        if (line[i] == '*' && (i+1 >= len || line[i+1] != '*')) {
            size_t end = line.find('*', i+1);
            if (end != std::string::npos && end > i+1) {
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

static void render_markdown(const std::string &text) {
    std::istringstream ss(text);
    std::string line;
    bool in_code = false;
    while (std::getline(ss, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.size() >= 3 && line.substr(0,3) == "```") {
            if (!in_code) {
                in_code = true;
                std::string lang = line.size() > 3 ? line.substr(3) : "code";
                while (!lang.empty() && lang[0]==' ') lang.erase(0,1);
                if (lang.empty()) lang = "code";
                std::cout << C_GRAY << "\xe2\x94\x8c\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80 " << lang << " \xe2\x94\x80\xe2\x94\x80\xe2\x94\x80" << C_RESET << std::endl;
            } else {
                in_code = false;
                std::cout << C_GRAY << "\xe2\x94\x94\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80" << C_RESET << std::endl;
            }
            continue;
        }
        if (in_code) {
            std::cout << C_GRAY << "\xe2\x94\x82 " << C_WHITE << line << C_RESET << std::endl;
            continue;
        }
        if (line.size()>=4 && line.substr(0,4)=="### ") { std::cout << C_H3 << "  \xe2\x96\xb8 " << line.substr(4) << C_RESET << std::endl; continue; }
        if (line.size()>=3 && line.substr(0,3)=="## ")  { std::cout << C_H2 << " \xe2\x96\xb8 " << line.substr(3) << C_RESET << std::endl; continue; }
        if (line.size()>=2 && line.substr(0,2)=="# ")   { std::cout << C_H1 << "\xe2\x96\xb8 " << line.substr(2) << C_RESET << std::endl; continue; }
        if (!line.empty() && line[0]=='>') {
            std::string c = line.size()>1 ? line.substr(1) : "";
            if (!c.empty() && c[0]==' ') c.erase(0,1);
            std::cout << C_QUOTE << "  \xe2\x94\x83 " << c << C_RESET << std::endl;
            continue;
        }
        if (line.size()>=3) {
            bool hr = true; char ch = line[0];
            if (ch=='-'||ch=='*'||ch=='_') { for (char x:line) if(x!=ch&&x!=' '){hr=false;break;} } else hr=false;
            if (hr) { std::cout << C_GRAY << "  \xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80" << C_RESET << std::endl; continue; }
        }
        if (line.size()>=2 && (line[0]=='-'||line[0]=='*') && line[1]==' ') {
            std::cout << C_BULLET << "  \xe2\x80\xa2 " << C_RESET << render_inline_md(line.substr(2)) << std::endl;
            continue;
        }
        { size_t p=0; while(p<line.size()&&line[p]>='0'&&line[p]<='9') ++p;
          if (p>0&&p<line.size()&&line[p]=='.'&&p+1<line.size()&&line[p+1]==' ') {
            std::cout << C_BULLET << "  " << line.substr(0,p) << ". " << C_RESET << render_inline_md(line.substr(p+2)) << std::endl;
            continue;
          }
        }
        if (line.size()>=4) {
            size_t sp=0; while(sp<line.size()&&line[sp]==' ') ++sp;
            if (sp>=2 && sp<line.size() && (line[sp]=='-'||line[sp]=='*') && sp+1<line.size() && line[sp+1]==' ') {
                std::string ind(sp/2, ' ');
                std::cout << C_BULLET << "  " << ind << "\xe2\x97\xa6 " << C_RESET << render_inline_md(line.substr(sp+2)) << std::endl;
                continue;
            }
        }
        std::cout << render_inline_md(line) << std::endl;
    }
    if (in_code) std::cout << C_GRAY << "\xe2\x94\x94\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80" << C_RESET << std::endl;
}


// ─────────────────────────── Сигналы ─────────────────────────
void signal_handler(int /*sig*/) {
    g_exit_requested = 1;
    rl_done = 1;
}

void do_exit() {
    std::cout << "\n" << C_YELLOW << "[Сохраняю историю перед выходом...]" << C_RESET << std::endl;
    save_history();
    write_history(READLINE_HIST_FILE.c_str());
    std::cout << C_YELLOW << "[Выход.]" << C_RESET << std::endl;
    exit(0);
}

// ─────────────────────────── Справка ─────────────────────────
void print_help() {
    std::cout << C_YELLOW
        << "Специальные команды:\n"
        << "  /save              — сохранить историю\n"
        << "  /load              — загрузить историю\n"
        << "  /clear             — очистить историю диалога\n"
        << "  /history           — показать историю\n"
        << "  /delete N          — удалить сообщение N из истории\n"
        << "  /retry             — повторить последний запрос\n"
        << "  /tokens            — показать использование токенов\n"
        << "  /model [name]      — показать/сменить модель\n"
        << "  /temp [0.0-2.0]    — показать/сменить температуру\n"
        << "  /maxtokens [N]     — показать/сменить max_tokens\n"
        << "  /system            — показать системный промпт\n"
        << "  /help              — эта справка\n"
        << "  /exit              — выход\n"
        << "\nМногострочный ввод:\n"
        << "  Enter              — новая строка (продолжение ввода)\n"
        << "  //                 — отправить сообщение (конец ввода)\n"
        << "  Пустая строка      — отправить однострочное сообщение\n"
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
            std::cout << C_MAGENTA << "[" << i << "] system: "    << C_RESET << cont << std::endl;
        else if (role == "user")
            std::cout << C_GREEN   << "[" << i << "] user: "      << C_RESET << cont << std::endl;
        else if (role == "assistant")
            std::cout << C_CYAN    << "[" << i << "] assistant: " << C_RESET << cont << std::endl;
        else
            std::cout << C_YELLOW  << "[" << i << "] " << role << ": " << C_RESET << cont << std::endl;
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
        // Защита от удаления системного промпта
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

// ─────────────────────────── API запрос ──────────────────────
// do_api_request — streaming version, returns full content or empty on error
// Also updates G.total_prompt_tokens / G.total_completion_tokens
// Returns special JSON-like string for backward compat or empty
std::string do_api_request() {
    std::string api_key = get_api_key();
    if (api_key.empty()) return "";

    CURL *curl = curl_easy_init();
    if (!curl) return "";

    trim_messages_if_needed();

    json jData = {
        {"model",       G.model},
        {"messages",    G.messages},
        {"temperature", G.temperature},
        {"max_tokens",  G.max_tokens},
        {"stream",      true}
    };
    std::string jsonData = jData.dump(-1, ' ', false, json::error_handler_t::replace);

    struct curl_slist *headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, "Accept: text/event-stream");
    std::string auth = "Authorization: Bearer " + api_key;
    headers = curl_slist_append(headers, auth.c_str());

    StreamContext sctx;

    curl_easy_setopt(curl, CURLOPT_URL, "https://openrouter.ai/api/v1/chat/completions");
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS,    jsonData.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)jsonData.size());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER,    headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, StreamCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA,     &sctx);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT,       120L);

    CURLcode res = curl_easy_perform(curl);

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        std::cerr << C_RED << "curl failed: " << curl_easy_strerror(res) << C_RESET << std::endl;
        return "";
    }

    // При ошибке HTTP стриминг мог не начаться — sctx.full_content содержит тело ошибки
    if (http_code != 200) {
        // Попробуем показать ошибку
        std::string err_text = sctx.full_content.empty() ? sctx.buffer : sctx.full_content;
        std::cerr << C_RED << "[HTTP " << http_code << "] "
                  << err_text.substr(0, std::min((size_t)500, err_text.size()))
                  << C_RESET << std::endl;
        return "";
    }

    // Стриминг уже вывел сырой текст — перерисовываем красиво через render_markdown
    if (!sctx.full_content.empty()) {
        struct winsize w;
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
        int term_cols = (w.ws_col > 0) ? w.ws_col : 80;
        int visual_lines = 0;
        int col = 0;
        for (char c : sctx.full_content) {
            if (c == '\n') { visual_lines++; col = 0; }
            else { col++; if (col >= term_cols) { visual_lines++; col = 0; } }
        }
        if (col > 0) visual_lines++;
        int lines_up = visual_lines + 2;
        std::cout << "\033[" << lines_up << "A\033[J";
        std::cout << C_BOLD << C_CYAN << "[Ассистент]:" << C_RESET << "\n";
        render_markdown(sctx.full_content);
        std::cout << std::endl;
    }

    // Обновляем счётчики токенов
    G.total_prompt_tokens     += sctx.prompt_tokens;
    G.total_completion_tokens += sctx.completion_tokens;

    // Возвращаем специальный маркер — content уже выведен на экран
    // Формируем pseudo-JSON для process_api_response
    if (sctx.full_content.empty()) return "";

    json result = {
        {"choices", json::array({{{"message", {{"content", sctx.full_content}}}}})}
    };
    return result.dump();
}

// ─────────────────────────── Обработка ответа ────────────────
bool process_api_response(const std::string &rawResponse) {
    if (rawResponse.empty()) {
        std::cerr << C_RED << "[Пустой ответ API]" << C_RESET << std::endl;
        return false;
    }

    try {
        json j = json::parse(rawResponse);
        if (!j.contains("choices") || j["choices"].empty()) {
            std::cerr << C_RED << "API error: "
                      << rawResponse.substr(0, std::min((size_t)300, rawResponse.size()))
                      << C_RESET << std::endl;
            return false;
        }

        std::string content = j["choices"][0]["message"]["content"];
        // Контент уже выведен стримингом, не дублируем

        // Парсинг bash-команды (content НЕ модифицируется)
        std::string cmd_result = for_com_parse(content);

        // Сохраняем оригинальный ответ ассистента (без результата команды)
        G.messages.push_back({{"role", "assistant"}, {"content", content}});

        if (!cmd_result.empty()) {
            G.messages.push_back({{"role", "user"},
                {"content", "[Результат выполнения команды]:\n" + cmd_result}});

            // Второй запрос после выполнения команды
            std::string raw2 = do_api_request();
            if (!raw2.empty()) {
                // Рекурсивно обрабатываем (с защитой от бесконечной рекурсии)
                // Но здесь делаем только один дополнительный уровень
                try {
                    json j2 = json::parse(raw2);
                    if (j2.contains("choices") && !j2["choices"].empty()) {
                        std::string content2 = j2["choices"][0]["message"]["content"];
                        // Контент уже выведен стримингом

                        std::string cmd_result2 = for_com_parse(content2);
                        G.messages.push_back({{"role", "assistant"}, {"content", content2}});

                        if (!cmd_result2.empty()) {
                            G.messages.push_back({{"role", "user"},
                                {"content", "[Результат выполнения команды]:\n" + cmd_result2}});
                        }
                    }
                } catch (const json::exception &e) {
                    std::cerr << C_RED << "JSON error (2nd response): " << e.what() << C_RESET << std::endl;
                }
            }
        }

        // Автосохранение каждые ~10 сообщений
        if (G.messages.size() >= 10 && G.messages.size() % 10 < 3) {
            save_history();
        }

    } catch (const json::exception &e) {
        std::cerr << C_RED << "JSON error: " << e.what() << C_RESET << std::endl;
        return false;
    }
    return true;
}

// ─────────────────────────── Ввод пользователя ───────────────
static bool get_user_input(std::string &out) {
    std::string result;
    bool first_line = true;
    int  line_num   = 1;

    while (true) {
        if (g_exit_requested) return false;

        std::string prompt = first_line
            ? "\n\001\033[1m\033[32m\002> \001\033[0m\002"
            : ("\001\033[32m\002" + std::to_string(line_num) + "> \001\033[0m\002");

        char *line = readline(prompt.c_str());

        if (!line) {
            if (!result.empty()) {
                out = result;
                add_history(result.size() <= 500
                    ? result.c_str()
                    : (result.substr(0, 500) + "...").c_str());
                return true;
            }
            return false;
        }

        std::string sline = sanitize_utf8(std::string(line));
        free(line);

        if (sline == "//") {
            if (result.empty()) {
                std::cout << C_GRAY
                          << "[Нет текста для отправки. Введите сообщение или /exit для выхода]"
                          << C_RESET << std::endl;
                first_line = true;
                line_num   = 1;
                result.clear();
                continue;
            }
            break;
        }

        if (first_line && sline.empty()) {
            out = "";
            return true;
        }

        if (!sline.empty() || !first_line) {
            if (!first_line) result += "\n";
            result += sline;

            if (first_line) {
                std::cout << C_GRAY
                          << "[Многострочный режим: Enter — новая строка, '//' — отправить]"
                          << C_RESET << std::endl;
            }
            first_line = false;
            line_num++;
        }
    }

    if (!result.empty()) {
        std::string hist = result.size() > 500 ? result.substr(0, 500) + "..." : result;
        add_history(hist.c_str());
    }

    out = result;
    return true;
}

// ─────────────────── Проверка команды с разделителем ─────────
// Возвращает true если userAnswer начинается с cmd и после cmd идёт пробел или конец строки
static bool match_command(const std::string &userAnswer, const std::string &cmd) {
    if (userAnswer.size() < cmd.size()) return false;
    if (userAnswer.substr(0, cmd.size()) != cmd) return false;
    if (userAnswer.size() == cmd.size()) return true;
    return userAnswer[cmd.size()] == ' ';
}

// Возвращает аргумент после команды (после пробела), или пустую строку
static std::string command_arg(const std::string &userAnswer, const std::string &cmd) {
    if (userAnswer.size() <= cmd.size() + 1) return "";
    return userAnswer.substr(cmd.size() + 1);
}

// ─────────────────────────── main ────────────────────────────
int main(int argc, char *argv[]) {
    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);

    // Загрузка системного промпта
    G.sys_prompt = load_system_prompt();
    if (G.sys_prompt.empty()) {
        G.sys_prompt =
            "То, что ты выведешь после ```bash будет сразу исполняться в системе через функцию system();. "
            "Используй максимально аккуратно, чтобы не навредить системе !!! "
            "Всегда придерживайся правила: только одна вставка на bash может быть в твоем ответе. "
            "Все инструкции, что указаны здесь выше ты должен постоянно помнить и не нарушать. "
            "ЭТО ВАЖНО! Результат выполнения команды будет добавлен к твоему сообщению автоматически. "
            "Таблицы оформлять табами (TAB), а не markdown-таблицами, т.к. вертикальные столбцы смещаются из-за шрифта. "
            "В папке ~/tmp возможно будет файл memo.md это твоя память. "
            "Если необходимо сделать запись в memo.md, то сохраняй самое важное, максимум три - пять строк, ДОПИСЫВАЯ в файл.";
    }
    // Системный промпт с правильной ролью "system"
    G.messages.push_back({{"role", "system"}, {"content", G.sys_prompt}});

    curl_global_init(CURL_GLOBAL_ALL);

    // ── Режим пайпа: stdin не терминал или есть аргументы ──
    bool pipe_mode = !isatty(fileno(stdin));
    bool has_args  = argc > 1;

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
        std::string raw = do_api_request();
        std::cout << std::endl;
        curl_global_cleanup();
        return 0;
    }

    // ── Интерактивный режим ──
    std::cout << C_BOLD << C_CYAN << "=== Chat CLI ===" << C_RESET << std::endl;
    std::cout << C_YELLOW << "Модель: " << G.model << C_RESET << std::endl;
    std::cout << C_YELLOW << "Введите /help для справки" << C_RESET << std::endl;
    std::cout << C_GRAY   << "Подсказка: Enter — новая строка, '//' — отправить сообщение"
              << C_RESET << std::endl;

    using_history();
    read_history(READLINE_HIST_FILE.c_str());

    // ── Главный цикл ──
    while (true) {
        if (g_exit_requested) do_exit();

        // ── Ввод пользователя ──
        std::string userAnswer;
        if (!get_user_input(userAnswer)) do_exit();

        if (g_exit_requested) do_exit();

        if (userAnswer.empty()) userAnswer = "(пустой ввод)";

        // ── Специальные команды ──
        if (userAnswer == "/help")    { print_help();    continue; }
        if (userAnswer == "/save")    { save_history();  continue; }
        if (userAnswer == "/load")    { load_history();  continue; }
        if (userAnswer == "/history") { print_history(); continue; }
        if (userAnswer == "/tokens")  { print_tokens();  continue; }
        if (userAnswer == "/exit")    { do_exit(); }
        if (userAnswer == "/system")  {
            std::cout << C_MAGENTA << "[Системный промпт]:\n" << G.sys_prompt << C_RESET << std::endl;
            continue;
        }
        if (userAnswer == "/retry") {
            if (!G.messages.empty() && G.messages.back()["role"] == "assistant") {
                G.messages.pop_back();
                std::cout << C_YELLOW << "[Повтор запроса...]" << C_RESET << std::endl;
                // Не добавляем новое сообщение, сразу отправляем запрос
            } else {
                std::cout << C_GRAY << "[Нет ответа ассистента для повтора]" << C_RESET << std::endl;
                continue;
            }
        } else if (userAnswer == "/clear") {
            G.messages.clear();
            G.messages.push_back({{"role", "system"}, {"content", G.sys_prompt}});
            G.total_prompt_tokens     = 0;
            G.total_completion_tokens = 0;
            std::cout << C_YELLOW << "[История очищена]" << C_RESET << std::endl;
            continue;
        } else if (match_command(userAnswer, "/delete")) {
            cmd_delete(command_arg(userAnswer, "/delete"));
            continue;
        } else if (match_command(userAnswer, "/model")) {
            std::string arg = command_arg(userAnswer, "/model");
            if (!arg.empty()) {
                G.model = arg;
                std::cout << C_YELLOW << "[Модель изменена на: " << G.model << "]" << C_RESET << std::endl;
            } else {
                std::cout << C_YELLOW << "[Текущая модель: " << G.model << "]" << C_RESET << std::endl;
            }
            continue;
        } else if (match_command(userAnswer, "/temp")) {
            std::string arg = command_arg(userAnswer, "/temp");
            if (!arg.empty()) {
                try {
                    double t = std::stod(arg);
                    if (t >= 0.0 && t <= 2.0) {
                        G.temperature = t;
                        std::cout << C_YELLOW << "[Температура: " << G.temperature << "]" << C_RESET << std::endl;
                    } else {
                        std::cerr << C_RED << "[Температура должна быть 0.0–2.0]" << C_RESET << std::endl;
                    }
                } catch (...) {
                    std::cerr << C_RED << "[Неверное значение]" << C_RESET << std::endl;
                }
            } else {
                std::cout << C_YELLOW << "[Температура: " << G.temperature << "]" << C_RESET << std::endl;
            }
            continue;
        } else if (match_command(userAnswer, "/maxtokens")) {
            std::string arg = command_arg(userAnswer, "/maxtokens");
            if (!arg.empty()) {
                try {
                    int mt = std::stoi(arg);
                    if (mt > 0) {
                        G.max_tokens = mt;
                        std::cout << C_YELLOW << "[max_tokens: " << G.max_tokens << "]" << C_RESET << std::endl;
                    } else {
                        std::cerr << C_RED << "[max_tokens должен быть > 0]" << C_RESET << std::endl;
                    }
                } catch (...) {
                    std::cerr << C_RED << "[Неверное значение]" << C_RESET << std::endl;
                }
            } else {
                std::cout << C_YELLOW << "[max_tokens: " << G.max_tokens << "]" << C_RESET << std::endl;
            }
            continue;
        } else if (userAnswer[0] == '/' && userAnswer != "(пустой ввод)") {
            // Неизвестная команда
            std::cerr << C_RED << "[Неизвестная команда: " << userAnswer << ". Введите /help]" << C_RESET << std::endl;
            continue;
        } else {
            // Обычное сообщение — добавляем в историю
            G.messages.push_back({{"role", "user"}, {"content", userAnswer}});
        }

        // ── API запрос ──
        std::cout << C_YELLOW << "[Запрос к API... модель: " << G.model
                  << ", temp: " << G.temperature
                  << ", max_tokens: " << G.max_tokens << "]" << C_RESET << std::endl;
        std::string rawResponse = do_api_request();

        if (g_exit_requested) do_exit();

        process_api_response(rawResponse);
    }

    curl_global_cleanup();
    return 0;
}
