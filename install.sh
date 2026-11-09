#!/bin/bash
set -e

# ─────────────────────────── Цвета ───────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RESET='\033[0m'

REPO_URL="https://raw.githubusercontent.com/swarik/Chat-Assist/main"
INSTALL_DIR="$HOME/.local/bin"
TMP_DIR="$(mktemp -d)"
NLOHMANN_DIR="$HOME/.local/include/nlohmann"

# Termux (Android) — определяем через $PREFIX
IS_TERMUX=0
if [ -n "$PREFIX" ] && [ -x "$PREFIX/bin/pkg" ]; then
    IS_TERMUX=1
fi

# C++ компилятор: g++ если есть, иначе clang++
CXX="${CXX:-}"
if [ -z "$CXX" ]; then
    if command -v g++ &>/dev/null; then CXX=g++
    elif command -v clang++ &>/dev/null; then CXX=clang++
    else CXX=g++
    fi
fi

# sudo нужен только для установки пакетов. В "curl | bash" stdin занят скриптом,
# поэтому без tty и без настроенного sudo пароль спросить негде — тогда пакеты не трогаем.
SUDO=""
if [ "$(id -u)" -ne 0 ] && [ "$IS_TERMUX" != "1" ]; then
    if command -v sudo >/dev/null 2>&1; then
        if sudo -n true 2>/dev/null; then
            SUDO="sudo"
        elif [ -t 0 ]; then
            SUDO="sudo"
        fi
    fi
fi

echo -e "${CYAN}=== sw_chat installer ===${RESET}"

# ─────────────────────────── Определяем пакетный менеджер ────
detect_pkg_manager() {
    if command -v apt-get &>/dev/null; then
        echo "apt"
    elif command -v dnf &>/dev/null; then
        echo "dnf"
    elif command -v pacman &>/dev/null; then
        echo "pacman"
    elif command -v zypper &>/dev/null; then
        echo "zypper"
    else
        echo "unknown"
    fi
}

# ─────────────────────────── Установка зависимостей ──────────
have_build_deps() {
    command -v "$CXX" >/dev/null 2>&1 || return 1
    local f found
    found=0
    for f in /usr/include/readline/readline.h /usr/local/include/readline/readline.h \
             "$HOME/.local/include/readline/readline.h"; do
        [ -f "$f" ] && { found=1; break; }
    done
    [ "$found" = 1 ] || return 1
    found=0
    for f in /usr/include/curl/curl.h /usr/include/*/curl/curl.h \
             /usr/local/include/curl/curl.h "$HOME/.local/include/curl/curl.h"; do
        [ -f "$f" ] && { found=1; break; }
    done
    [ "$found" = 1 ] || return 1
    return 0
}

install_deps() {
    local pm="$1"
    if [ "$IS_TERMUX" = "1" ]; then
        echo -e "${YELLOW}[Устанавливаю зависимости (Termux)...]${RESET}"
        pkg install -y clang readline libcurl make
        echo -e "${GREEN}[Зависимости (Termux) установлены]${RESET}"
        return
    fi
    if have_build_deps; then
        echo -e "${GREEN}[Зависимости уже есть — установку пакетов пропускаю]${RESET}"
        return
    fi
    if [ -z "$SUDO" ] && [ "$(id -u)" -ne 0 ]; then
        echo -e "${RED}[Не хватает зависимостей, а установить их без пароля нельзя]${RESET}"
        echo "   Поставьте вручную и запустите снова:"
        echo "     Debian/Ubuntu: sudo apt-get install -y g++ libreadline-dev libcurl4-openssl-dev"
        echo "     Fedora/RHEL:   sudo dnf install -y gcc-c++ readline-devel libcurl-devel"
        echo "     Arch:          sudo pacman -S --noconfirm gcc readline curl"
        echo "     openSUSE:      sudo zypper install -y gcc-c++ readline-devel libcurl-devel"
        exit 1
    fi
    echo -e "${YELLOW}[Устанавливаю зависимости...]${RESET}"
    case "$pm" in
        apt)
            $SUDO apt-get update -qq
            $SUDO apt-get install -y g++ libreadline-dev libcurl4-openssl-dev
            ;;
        dnf)
            $SUDO dnf install -y gcc-c++ readline-devel libcurl-devel
            ;;
        pacman)
            $SUDO pacman -Sy --noconfirm gcc readline curl
            ;;
        zypper)
            $SUDO zypper install -y gcc-c++ readline-devel libcurl-devel
            ;;
        *)
            echo -e "${RED}[Неизвестный пакетный менеджер. Установите вручную: g++, libreadline-dev, libcurl-dev]${RESET}"
            exit 1
            ;;
    esac
    echo -e "${GREEN}[Зависимости установлены]${RESET}"
}

# ─────────────────────────── nlohmann/json (header-only) ─────
install_nlohmann() {
    mkdir -p "$NLOHMANN_DIR"
    # Правка 3: не перезаписываем уже установленный json.hpp — у пользователя
    # может быть более новая версия (системная или из другого проекта).
    if [ -s "$NLOHMANN_DIR/json.hpp" ]; then
        echo -e "${GREEN}[nlohmann/json уже установлен: $NLOHMANN_DIR/json.hpp — пропускаю]${RESET}"
        return
    fi
    echo -e "${YELLOW}[Устанавливаю nlohmann/json...]${RESET}"
    # Прибито к стабильному тегу — develop-ветка ломает сборки
    curl -fsSL "https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp" \
        -o "$NLOHMANN_DIR/json.hpp"
    # Правка 4: curl может вернуть 0, но оставить пустой файл (прокси, DNS-хайджек).
    if [ ! -s "$NLOHMANN_DIR/json.hpp" ]; then
        echo -e "${RED}[Не удалось скачать json.hpp — файл пуст или отсутствует]${RESET}"
        exit 1
    fi
    echo -e "${GREEN}[nlohmann/json установлен: $NLOHMANN_DIR/json.hpp]${RESET}"
}

# ─────────────────────────── Скачиваем исходник ───────────────
download_source() {
    echo -e "${YELLOW}[Скачиваю sw_chat.cpp...]${RESET}"
    curl -fsSL "$REPO_URL/sw_chat.cpp" -o "$TMP_DIR/sw_chat.cpp"
    if [ ! -s "$TMP_DIR/sw_chat.cpp" ]; then
        echo -e "${RED}[Файл не скачан или пуст: $REPO_URL/sw_chat.cpp]${RESET}"
        exit 1
    fi
    echo -e "${GREEN}[Исходник скачан ($(wc -c < "$TMP_DIR/sw_chat.cpp") байт)]${RESET}"
}

# ─────────────────────────── Компиляция ──────────────────────
compile() {
    echo -e "${YELLOW}[Компилирую...]${RESET}"
    "$CXX" -std=c++17 -O2 \
        -I"$HOME/.local/include" \
        -o "$TMP_DIR/sw_chat" \
        "$TMP_DIR/sw_chat.cpp" \
        -lreadline -lcurl -lpthread
    echo -e "${GREEN}[Компиляция успешна]${RESET}"
}

# ─────────────────────────── Установка бинарника ─────────────
install_binary() {
    if [ ! -x "$TMP_DIR/sw_chat" ]; then
        echo -e "${RED}[Бинарник не создан — компиляция не удалась]${RESET}"
        exit 1
    fi
    echo -e "${YELLOW}[Устанавливаю бинарник в $INSTALL_DIR...]${RESET}"
    mkdir -p "$INSTALL_DIR"
    # Правка 5: атомарная установка — копируем во временный файл рядом и mv.
    # Это защищает от битого рабочего бинаря, если cp упадёт на середине.
    local tmp_install="$INSTALL_DIR/.sw_chat.new.$$"
    if ! cp "$TMP_DIR/sw_chat" "$tmp_install"; then
        echo -e "${RED}[Не удалось скопировать бинарник во временный файл]${RESET}"
        rm -f "$tmp_install"
        exit 1
    fi
    chmod +x "$tmp_install"
    if ! mv -f "$tmp_install" "$INSTALL_DIR/sw_chat"; then
        echo -e "${RED}[Не удалось заменить $INSTALL_DIR/sw_chat]${RESET}"
        rm -f "$tmp_install"
        exit 1
    fi
    echo -e "${GREEN}[Установлено: $INSTALL_DIR/sw_chat]${RESET}"
}

# ─────────────────────────── PATH ────────────────────────────
check_path() {
    if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
        echo -e "${YELLOW}[Добавляю $INSTALL_DIR в PATH...]${RESET}"
        SHELL_RC=""
        if [ -f "$HOME/.bashrc" ]; then SHELL_RC="$HOME/.bashrc"
        elif [ -f "$HOME/.zshrc" ]; then SHELL_RC="$HOME/.zshrc"
        fi
        if [ -n "$SHELL_RC" ]; then
            # Правка 1: не дублируем запись при повторном запуске install.sh.
            if grep -qF '# sw_chat' "$SHELL_RC" 2>/dev/null; then
                echo -e "${GREEN}[Строка PATH уже есть в $SHELL_RC — пропускаю]${RESET}"
            else
                echo "" >> "$SHELL_RC"
                echo "# sw_chat" >> "$SHELL_RC"
                echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
                echo -e "${YELLOW}[Добавлено в $SHELL_RC. Выполните: source $SHELL_RC]${RESET}"
            fi
        else
            echo -e "${YELLOW}[Добавьте вручную в ваш shell rc: export PATH=\"\$HOME/.local/bin:\$PATH\"]${RESET}"
        fi
    fi
}

# ─────────────────────────── API ключ ────────────────────────
setup_api_key() {
    if [ -z "$302_API_KEY" ] && [ ! -f "$HOME/.config/302_key" ]; then
        echo -e "${YELLOW}[API ключ не найден]${RESET}"
        echo -e "Получите ключ на ${CYAN}https://302.ai${RESET}"
        # Правка 2: при запуске через pipe (curl ... | bash) stdin занят скриптом —
        # читаем из /dev/tty. Если tty недоступен — просто пропускаем шаг.
        apikey=""
        if [ -t 0 ]; then
            read -rp "Введите ваш 302.ai API ключ (или Enter чтобы пропустить): " apikey
        elif [ -r /dev/tty ]; then
            read -rp "Введите ваш 302.ai API ключ (или Enter чтобы пропустить): " apikey < /dev/tty
        else
            echo -e "${YELLOW}[Нет tty — пропускаю ввод. Сохраните ключ позже в ~/.config/302_key]${RESET}"
        fi
        if [ -n "$apikey" ]; then
            mkdir -p "$HOME/.config"
            printf "%s" "$apikey" > "$HOME/.config/302_key"
            chmod 600 "$HOME/.config/302_key"
            echo -e "${GREEN}[API ключ сохранён: ~/.config/302_key]${RESET}"
        else
            echo -e "${YELLOW}[Пропущено. Сохраните ключ в ~/.config/302_key или переменную 302_API_KEY]${RESET}"
        fi
    else
        echo -e "${GREEN}[API ключ найден]${RESET}"
    fi
}

# ─────────────────────────── Cleanup ─────────────────────────
cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

# ─────────────────────────── Главный сценарий ────────────────
PM=$(detect_pkg_manager)
echo -e "${CYAN}[Пакетный менеджер: $PM]${RESET}"

install_deps "$PM"
install_nlohmann
download_source
compile
install_binary
check_path
setup_api_key

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗${RESET}"
echo -e "${GREEN}║       sw_chat успешно установлен!        ║${RESET}"
echo -e "${GREEN}║  Запуск: sw_chat                         ║${RESET}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${RESET}"
