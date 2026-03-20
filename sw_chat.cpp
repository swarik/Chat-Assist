#include <iostream>
#include <sys/ioctl.h>
#include <unistd.h>
#include <fstream>
#include <cstdio>
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
static std::string get_home_dir() {
    const char* h = getenv("HOME");
    return h ? std::string(h) : "/tmp";
}

// ─────────────────────────── Константы ───────────────────────
#define CMD_TIMEOUT         150
#define MAX_CMD_OUTPUT      196000
#define MAX_MESSAGES        500
#define DEFAULT_TEMPERATURE 0.7
#define DEFAULT_MAX_TOKENS  52000

static std::string HISTORY_FILE;
static std::string SYSTEM_PROMPT_FILE;
static std::string READLINE_HIST_FILE;

// ─────────────────────────── Глобальное состояние ────────────
struct ChatSession {
    std::vector<json> messages;
    std::string       history_file;

    //std::string       model          = "nvidia/nemotron-3-super-120b-a12b:free";
    //std::string       model          = "minimax/minimax-m2.7";
    //std::string       model          = "anthropic/claude-sonnet-4";
    //std::string       model          = "openai/gpt-5.2";
    //std::string       model          = "google/gemini-3.1-pro-preview";
    //std::string       model          = "x-ai/grok-4";
    //std::string       model          = "qwen/qwen3-max-thinking";
    //std::string       model          = "xiaomi/mimo-v2-flash";
    //std::string       model          = "xiaomi/mimo-v2-pro"
    //std::string       model          = "nex-agi/deepseek-v3.1-nex-n1";
    //std::string       model          = "anthropic/claude-opus-4.6";
    std::string       model          = "anthropic/claude-sonnet-4.6";


    std::string       sys_prompt;
    double            temperature    = DEFAULT_TEMPERATURE;
    int               max_tokens     = DEFAULT_MAX_TOKENS;
    int               total_prompt_tokens     = 0;
    int               total_completion_tokens = 0;
    bool              autorun                 = false;
};

static ChatSession G;

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
        rl_done = 1;
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
static std::vector<size_t> find_table_cols(const std::string &line) {
    std::vector<size_t> cols;
    for (size_t i = 0; i < line.size(); ++i)
        if (line[i] == '|') cols.push_back(i);
    return cols;
}

static void render_table_row(const std::string &line, const std::vector<size_t> &cols) {
    std::cout << "│";
    size_t pos = 0;
    for (size_t i = 0; i < cols.size(); ++i) {
        size_t end = (i + 1 < cols.size()) ? cols[i] : line.size();
        std::string cell = line.substr(pos, end - pos);
        // Убираем | и пробелы
        while (!cell.empty() && (cell[0]=='|' || cell[0]==' ')) cell.erase(cell.begin(), cell.begin()+1);
        while (!cell.empty() && (cell.back()=='|' || cell.back()==' ')) cell.pop_back();
        
        // Выравнивание
        std::cout << " " << render_inline_md(cell);
        size_t pad = (i + 1 < cols.size()) ? (cols[i+1] - cols[i] - 1) : 10;
        if (pad > cell.size() + 1) {
            for (size_t p = 0; p < pad - cell.size() - 1; ++p) std::cout << " ";
        }
    }
    std::cout << "│\n";
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

        // ── Markdown таблицы ──
        if (line.find('|') != std::string::npos) {
            bool is_separator = true;
            for (char c : line) if (c!='|' && c!='-' && c!=' ' && c!=':') { is_separator = false; break; }
            
            if (is_separator) {
                // Разделитель ────
                std::cout << C_GRAY;
                for (char c : line) {
                    if (c == '-') std::cout << "─";
                    else if (c == ':') std::cout << ":";
                    else std::cout << " ";
                }
                std::cout << C_RESET << "\n";
                continue;
            }
            
            // Строка таблицы
            auto cols = find_table_cols(line);
            if (cols.size() >= 2) {
                render_table_row(line, cols);
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
void save_history() {
    try {
        json j = json::array();
        for (auto &m : G.messages) j.push_back(m);
        std::ofstream f(G.history_file);
        if (f.is_open()) {
            f << j.dump(2, ' ', false, json::error_handler_t::replace);
            std::cout << C_YELLOW << "[История сохранена: " << G.history_file
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
        // Проверка на пустой файл
        std::string content((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
        if (content.empty()) return false;
        
        json j = json::parse(content);
        G.messages.clear();
        for (auto &m : j) G.messages.push_back(m);
        if (G.messages.empty() || G.messages[0]["role"] != "system") {
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

    if (!G.autorun) {
        char *rl_confirm = readline(C_YELLOW "[Выполнить команду? (y/n|д/н)]: " C_RESET);
        if (!rl_confirm) return "";
        std::string confirm(rl_confirm);
        free(rl_confirm);
        if (confirm != "y" && confirm != "Y" && confirm != "д" && confirm != "Д") {
            std::cout << C_RED << "[Выполнение отменено пользователем]" << C_RESET << std::endl;
            return "";
        }
    } else {
        std::cout << C_YELLOW << "[Autorun: выполняю автоматически]" << C_RESET << std::endl;
    }
    // БАГ 8: предупреждаем если в ответе несколько bash-блоков (выполняется только первый)
    {
        size_t second = content.find("```bash", cnt_start + open_tag.size());
        if (second != std::string::npos) {
            std::cout << C_YELLOW
                      << "[Внимание: в ответе несколько bash-блоков, выполняется только первый]"
                      << C_RESET << std::endl;
        }
    }
    std::cout << C_YELLOW << "[Выполняю...]" << C_RESET << std::endl;
    std::string result = exec_with_timeout(com_cont, CMD_TIMEOUT);
    std::cout << C_BLUE << "[Результат]:\n" << result << C_RESET << std::endl;
    return result;
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

// ─────────────────────────── Streaming SSE ───────────────────
struct StreamContext {
    std::string buffer;
    std::string full_content;
    int         prompt_tokens     = 0;
    int         completion_tokens = 0;
    bool        done              = false;
};

static size_t StreamCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    if (g_stream_abort) return 0;

    StreamContext *ctx = static_cast<StreamContext*>(userp);
    std::string chunk((char*)contents, size * nmemb);
    ctx->buffer += chunk;

    std::string::size_type pos;
    while ((pos = ctx->buffer.find("\n")) != std::string::npos) {
        std::string line = ctx->buffer.substr(0, pos);
        ctx->buffer.erase(0, pos + 1);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (line == "data: [DONE]") { ctx->done = true; continue; }
        if (line.rfind("data: ", 0) != 0) continue;
        std::string json_str = line.substr(6);
        try {
            json j = json::parse(json_str);
            if (j.contains("choices") && !j["choices"].empty()) {
                auto &choice = j["choices"][0];
                if (choice.contains("delta") && choice["delta"].contains("content")) {
                    std::string token = choice["delta"]["content"];
                    if (!token.empty()) ctx->full_content += token;
                }
            }
            if (j.contains("usage")) {
                ctx->prompt_tokens     = j["usage"].value("prompt_tokens", 0);
                ctx->completion_tokens = j["usage"].value("completion_tokens", 0);
            }
        } catch (...) {}
    }
    return size * nmemb;
}

// ─────────────────────────── API запрос ──────────────────────
// Возвращает full_content или "" при ошибке/прерывании
// aborted — true если пользователь прервал Ctrl+C
std::string do_api_request(bool &aborted) {
    aborted = false;
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

    // Устанавливаем флаги под мьютексом
    {
        std::lock_guard<std::mutex> lock(g_stream_mutex);
        g_stream_abort = 0;
        g_in_streaming = 1;
    }

    // Запускаем спиннер
    g_spinner_stop.store(false);
    g_spinner_active.store(true);
    std::thread spinner_t(spinner_thread_func, G.model);

    curl_easy_setopt(curl, CURLOPT_URL, "https://openrouter.ai/api/v1/chat/completions");
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS,    jsonData.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)jsonData.size());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER,    headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, StreamCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA,     &sctx);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT,       220L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 15L);

    CURLcode res = curl_easy_perform(curl);

    // Останавливаем спиннер (немедленно)
    g_spinner_stop.store(true);
    g_spinner_active.store(false);
    if (spinner_t.joinable()) spinner_t.join();

    // Сбрасываем флаги под мьютексом
    {
        std::lock_guard<std::mutex> lock(g_stream_mutex);
        g_in_streaming = 0;
    }

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    // Проверяем прерывание пользователем под мьютексом
    bool was_aborted = false;
    {
        std::lock_guard<std::mutex> lock(g_stream_mutex);
        if (g_stream_abort) {
            was_aborted = true;
            g_stream_abort = 0;
        }
    }
    
    if (was_aborted) {
        aborted = true;
        std::cout << "\n" << C_YELLOW << "[Вывод прерван пользователем]" << C_RESET << std::endl;
        // Возвращаем то что успело накопиться (может быть частичный ответ)
        return sctx.full_content;
    }

    if (res != CURLE_OK) {
        std::cerr << C_RED << "curl failed: " << curl_easy_strerror(res) << C_RESET << std::endl;
        return "";
    }

    if (http_code != 200) {
        std::string err_text = sctx.full_content + sctx.buffer;
        if (err_text.empty()) err_text = "(empty response)";
        std::cerr << C_RED << "[HTTP " << http_code << "] "
                  << err_text.substr(0, std::min((size_t)500, err_text.size()))
                  << C_RESET << std::endl;
        return "";
    }

    G.total_prompt_tokens     += sctx.prompt_tokens;
    G.total_completion_tokens += sctx.completion_tokens;

    return sanitize_utf8(sctx.full_content);
}

// ─────────────────────────── Обработка ответа ────────────────
// Выводит ответ красиво, обрабатывает bash-команды
// aborted — если ответ был прерван, не добавляем его в историю
void process_response(const std::string &content, bool aborted, size_t msgs_before = 0) {
    if (content.empty()) return;

    // Красивый вывод
    std::cout << "\n" << C_BOLD << C_CYAN << "[Ассистент]:" << C_RESET << "\n";
    render_markdown(content);
    std::cout << std::endl;

    if (aborted) {
        char *rl_ans = readline(C_YELLOW "[Ответ был прерван. Сохранить частичный ответ в историю? (y/n)]: " C_RESET);
        std::string ans;
        if (rl_ans) { ans = std::string(rl_ans); free(rl_ans); }
        if (ans != "y" && ans != "Y" && ans != "д" && ans != "Д") {
            if (msgs_before > 0 && msgs_before <= G.messages.size()) {
                G.messages.resize(msgs_before);
                std::cout << C_GRAY << "[История откачена к состоянию до запроса]"
                          << C_RESET << std::endl;
            } else if (!G.messages.empty() && G.messages.back()["role"] == "user") {
                G.messages.pop_back();
                std::cout << C_GRAY << "[Вопрос и частичный ответ удалены из истории]"
                          << C_RESET << std::endl;
            }
            return;
        }
    }

    // Сохраняем ответ ассистента в историю
    G.messages.push_back({{"role", "assistant"}, {"content", content}});

    if (aborted) return;

    // Цикл выполнения bash-команд из ответов (до MAX_BASH_CHAIN итераций)
    const int MAX_BASH_CHAIN = 7;
    std::string current_content = content;
    for (int chain = 0; chain < MAX_BASH_CHAIN; ++chain) {
        std::string cmd_result = for_com_parse(current_content);
        if (cmd_result.empty()) break;

        G.messages.push_back({{"role", "user"},
            {"content", "[Результат выполнения команды]:\n" + cmd_result}});

        bool chain_aborted = false;
        std::string next_content = do_api_request(chain_aborted);
        if (next_content.empty()) break;

        std::cout << "\n" << C_BOLD << C_CYAN << "[Ассистент]:" << C_RESET << "\n";
        render_markdown(next_content);
        std::cout << std::endl;

        if (chain_aborted) {
            std::cout << C_YELLOW << "[Ответ прерван, не сохранён]"
                      << C_RESET << std::endl;
            break;
        }

        G.messages.push_back({{"role", "assistant"}, {"content", next_content}});
        current_content = next_content;
    }

    // Автосохранение каждые 10 сообщений
    if (G.messages.size() >= 10 && G.messages.size() % 10 == 0) {
        save_history();
    }
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
    
    std::cout << "\n" << C_YELLOW << "[Сохраняю историю перед выходом...]" << C_RESET << std::endl;
    save_history();
    write_history(READLINE_HIST_FILE.c_str());
    curl_global_cleanup();
    std::cout << C_YELLOW << "[Выход.]" << C_RESET << std::endl;
    exit(0);
}

// ─────────────────────────── Справка ─────────────────────────
// ─────────────────────── Список моделей ──────────────────────
static const std::vector<std::string> AVAILABLE_MODELS = {
    "anthropic/claude-opus-4.6",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "minimax/minimax-m2.7",
    "anthropic/claude-sonnet-4",
    "openai/gpt-5.2",
    "google/gemini-3.1-pro-preview",
    "x-ai/grok-4",
    "qwen/qwen3-max-thinking",
    "xiaomi/mimo-v2-flash",
    "nex-agi/deepseek-v3.1-nex-n1",
    "anthropic/claude-sonnet-4.6",
    "xiaomi/mimo-v2-pro",
};

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
        << "  /history           — показать историю\n"
        << "  /delete N          — удалить сообщение N из истории\n"
        << "  /retry             — повторить последний запрос\n"
        << "  /tokens            — показать использование токенов\n"
        << "  /model [name|N]    — выбор модели из списка / по имени / по номеру\n"
        << "  /temp [0.0-2.0]    — показать/сменить температуру\n"
        << "  /maxtokens [N]     — показать/сменить max_tokens\n"
        << "  /system            — показать системный промпт\n"
        << "  /file <path> [msg] — загрузить файл и задать вопрос\n"
        << "  /autorun           — вкл/выкл авто-выполнение bash\n"
        << "  /cost              — стоимость токенов в $\n"
        << "  /help              — эта справка\n"
        << "  /exit              — выход\n"
        << "\nМногострочный ввод:\n"
        << "  Enter              — новая строка (продолжение ввода)\n"
        << "  //                 — отправить сообщение (конец ввода)\n"
        << "  Пустая строка      — отправить однострочное сообщение\n"
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
    {"anthropic/claude-opus",      5.0,   25.0},
    {"anthropic/claude-sonnet",    3.0,   15.0},
    {"anthropic/claude-haiku",     0.25,  1.25},
    {"openai/gpt-5",               10.0,  30.0},
    {"openai/gpt-4.1",             2.0,   8.0},
    {"openai/gpt-4.1-mini",        0.4,   1.6},
    {"openai/gpt-4.1-nano",        0.1,   0.4},
    {"openai/o3",                  10.0,  40.0},
    {"openai/o4-mini",             1.1,   4.4},
    {"google/gemini-2.5-pro",      1.25,  10.0},
    {"google/gemini-2.5-flash",    0.15,   0.6},
    {"google/gemini-3",            1.25,  10.0},
    {"x-ai/grok-4",                10.0,  30.0},
    {"x-ai/grok-3",                3.0,   15.0},
    {"x-ai/grok-3-mini",           0.3,   0.5},
    {"deepseek/deepseek-r1",       0.55,  2.19},
    {"deepseek/deepseek-chat",     0.27,  1.10},
    {"qwen/qwen3",                 0.16,  0.64},
    {"meta-llama/llama-4",         0.2,   0.6},
    {"minimax/minimax-m2.7",       0.3,   1.2},
    {"xiaomi/mimo-v2-pro",         1.0,   3.0},
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

    std::cout << C_MAGENTA << "\n╔══════════════════════════════════════╗\n";
    std::cout << "║       Использование токенов          ║\n";
    std::cout << "╠══════════════════════════════════════╣\n";
    std::cout << "║  Модель:     " << G.model;
    // Паддинг
    int pad = 24 - (int)G.model.size();
    for (int i = 0; i < pad; ++i) std::cout << " ";
    if (pad < 0) std::cout << "\n║              ";
    std::cout << "║\n";
    std::cout << "║  Промпт:     ";
    printf("%8d токенов", G.total_prompt_tokens);
    std::cout << "       ║\n";
    std::cout << "║  Ответы:     ";
    printf("%8d токенов", G.total_completion_tokens);
    std::cout << "       ║\n";
    std::cout << "║  Всего:      ";
    printf("%8d токенов", G.total_prompt_tokens + G.total_completion_tokens);
    std::cout << "       ║\n";
    if (found) {
        std::cout << "╠══════════════════════════════════════╣\n";
        std::cout << "║  Промпт:     ";
        printf("$%-8.4f", prompt_cost);
        std::cout << "              ║\n";
        std::cout << "║  Ответы:     ";
        printf("$%-8.4f", completion_cost);
        std::cout << "              ║\n";
        std::cout << "║  Итого:      ";
        printf("$%-8.4f", total_cost);
        std::cout << "              ║\n";
        std::cout << "║  (";
        printf("$%.2f/$%.2f", p_price, c_price);
        std::cout << " за 1M токенов)     ║\n";
    } else {
        std::cout << "╠══════════════════════════════════════╣\n";
        std::cout << "║  Цены для модели не найдены          ║\n";
    }
    std::cout << "╚══════════════════════════════════════╝" << C_RESET << std::endl;
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

        char *line = readline(prompt.c_str());

        if (!line) {
            // EOF (Ctrl+D)
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
                      << "[Многострочный режим: Enter — новая строка, '//' — отправить]"
                      << C_RESET << std::endl;
        } else {
            // Пустая строка в многострочном режиме — отправляем
            if (sline.empty()) {
                break;
            }
            result += "\n";
            result += sline;
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
            "Всегда придерживайся правила: только одна вставка на bash может быть в твоем ответе. "
            "Все инструкции, что указаны здесь выше ты должен постоянно помнить и не нарушать. "
            "ЭТО ВАЖНО! Результат выполнения команды будет добавлено к твоему сообщению автоматически. "
            "Таблицы оформлять табами (TAB), а не markdown-таблицами, т.к. вертикальные столбцы смещаются из-за шрифта. "
            "В папке ~/tmp возможно будет файл memo.md это твоя память. "
            "Если необходимо сделать запись в memo.md, то сохраняй самое важное, максимум три - пять строк, ДОПИСЫВАЯ в файл.";
    }
    G.messages.push_back({{"role", "system"}, {"content", G.sys_prompt}});

    curl_global_init(CURL_GLOBAL_ALL);

    // ── Режим пайпа / аргументов ──
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
        bool aborted = false;
        std::string content = do_api_request(aborted);
        if (!content.empty()) {
            std::cout << "\n";
            render_markdown(content);
            std::cout << std::endl;
        }
        curl_global_cleanup();
        return 0;
    }

    // ── Интерактивный режим ──
    std::cout << C_BOLD << C_CYAN << "=== Chat CLI ===" << C_RESET << std::endl;
    std::cout << C_YELLOW << "Модель: " << G.model << C_RESET << std::endl;
    std::cout << C_YELLOW << "Введите /help для справки" << C_RESET << std::endl;
    std::cout << C_GRAY   << "Autorun: " << (G.autorun ? "вкл" : "выкл")
              << " (переключить: /autorun)" << C_RESET << std::endl;
    std::cout << C_GRAY   << "Подсказка: Enter — новая строка, '//' — отправить, "
                             "Ctrl+C во время ответа — прервать"
              << C_RESET << std::endl;

    using_history();
    read_history(READLINE_HIST_FILE.c_str());

    while (true) {
        if (g_exit_requested) do_exit();

        std::string userAnswer;
        if (!get_user_input(userAnswer)) do_exit();

        if (g_exit_requested) do_exit();

        if (userAnswer.empty()) continue; // пустой ввод — просто игнорируем

        // ── Специальные команды ──
        if (userAnswer == "/help")    { print_help();    continue; }
        if (userAnswer == "/save")    { save_history();  continue; }
        if (userAnswer == "/load")    { load_history();  continue; }
        if (userAnswer == "/history") { print_history(); continue; }
        if (userAnswer == "/tokens")  { print_tokens();  continue; }
        if (userAnswer == "/cost")    { print_cost();    continue; }
        if (userAnswer == "/autorun") {
            G.autorun = !G.autorun;
            std::cout << C_YELLOW << "[Autorun: "
                      << (G.autorun ? "ВКЛЮЧЁН ⚡" : "выключен")
                      << "]" << C_RESET << std::endl;
            continue;
        }
        if (userAnswer == "/exit")    { do_exit(); }
        if (userAnswer == "/system")  {
            std::cout << C_MAGENTA << "[Системный промпт]:\n"
                      << G.sys_prompt << C_RESET << std::endl;
            continue;
        }
        if (userAnswer == "/retry") {
            // БАГ 5: ищем последний assistant, затем откатываемся до user перед ним
            // чтобы удалить и assistant, и возможные bash-результаты после него
            int last_assistant = -1;
            for (int i = (int)G.messages.size() - 1; i >= 0; --i) {
                if (G.messages[i]["role"] == "assistant") {
                    last_assistant = i;
                    break;
                }
            }
            if (last_assistant > 0) {
                // Ищем user-сообщение перед этим assistant (это и есть наш вопрос)
                int user_before = -1;
                for (int i = last_assistant - 1; i >= 0; --i) {
                    if (G.messages[i]["role"] == "user") {
                        user_before = i;
                        break;
                    }
                }
                // Обрезаем до user_before (не включая его) — то есть до нашего вопроса
                // Сам вопрос остаётся, assistant и bash-результаты удаляются
                if (user_before >= 0) {
                    G.messages.resize(user_before + 1);
                } else {
                    G.messages.resize(last_assistant);
                }
                std::cout << C_YELLOW << "[Повтор запроса...]" << C_RESET << std::endl;
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
        std::string content = do_api_request(aborted);

        if (g_exit_requested) do_exit();

        process_response(content, aborted, msgs_before);
    }

    curl_global_cleanup();
    return 0;
}
