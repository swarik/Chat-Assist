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
install_deps() {
    local pm="$1"
    echo -e "${YELLOW}[Устанавливаю зависимости...]${RESET}"
    if [ "$IS_TERMUX" = "1" ]; then
        # Termux: без sudo, имена пакетов свои; заголовки readline/curl в базовых пакетах
        pkg install -y clang readline libcurl make
        echo -e "${GREEN}[Зависимости (Termux) установлены]${RESET}"
        return
    fi
    case "$pm" in
        apt)
            apt-get update -qq
            apt-get install -y g++ libreadline-dev libcurl4-openssl-dev
            ;;
        dnf)
            dnf install -y gcc-c++ readline-devel libcurl-devel
            ;;
        pacman)
            pacman -Sy --noconfirm gcc readline curl
            ;;
        zypper)
            zypper install -y gcc-c++ readline-devel libcurl-devel
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
    echo -e "${YELLOW}[Устанавливаю nlohmann/json...]${RESET}"
    mkdir -p "$NLOHMANN_DIR"
    # Прибито к стабильному тегу — develop-ветка ломает сборки
    curl -fsSL "https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp" \
        -o "$NLOHMANN_DIR/json.hpp"
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
    cp "$TMP_DIR/sw_chat" "$INSTALL_DIR/sw_chat"
    chmod +x "$INSTALL_DIR/sw_chat"
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
            echo "" >> "$SHELL_RC"
            echo "# sw_chat" >> "$SHELL_RC"
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
            echo -e "${YELLOW}[Добавлено в $SHELL_RC. Выполните: source $SHELL_RC]${RESET}"
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
        read -rp "Введите ваш 302.ai API ключ (или Enter чтобы пропустить): " apikey
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
