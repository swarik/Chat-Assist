#!/usr/bin/env bash
# sw_chat installer — скачивает исходник с GitHub, собирает и ставит в ~/.local/bin
# Использование:
#   curl -fsSL https://raw.githubusercontent.com/swarik/Chat-Assist/main/install.sh | bash
#   bash install.sh
#   bash install.sh --local
set -euo pipefail

RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[1;33m'
CYAN=$'\033[0;36m'
GRAY=$'\033[0;90m'
RESET=$'\033[0m'

GITHUB_USER="swarik"
GITHUB_REPO="Chat-Assist"
GITHUB_BRANCH="main"
REPO_RAW="https://raw.githubusercontent.com/${GITHUB_USER}/${GITHUB_REPO}/${GITHUB_BRANCH}"

INSTALL_DIR="${HOME}/.local/bin"
INCLUDE_DIR="${HOME}/.local/include"
NLOHMANN_DIR="${INCLUDE_DIR}/nlohmann"
CONFIG_DIR="${HOME}/.config"
KEY_FILE="${CONFIG_DIR}/302_key"
NLOHMANN_URL="https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp"

USE_LOCAL=0
SKIP_DEPS=0
for arg in "$@"; do
  case "$arg" in
    --local|-l) USE_LOCAL=1 ;;
    --skip-deps) SKIP_DEPS=1 ;;
    -h|--help)
      cat <<'EOF'
Установка sw_chat из GitHub

  bash install.sh              # скачать, собрать, поставить
  bash install.sh --local      # ./sw_chat.cpp рядом со скриптом
  bash install.sh --skip-deps  # не ставить пакеты через sudo

После установки: sw_chat
Ключ: ~/.config/302_key  или  export 302_API_KEY=...
EOF
      exit 0
      ;;
  esac
done

info()  { echo -e "${CYAN}==>${RESET} $*"; }
ok()    { echo -e "${GREEN}[ok]${RESET} $*"; }
warn()  { echo -e "${YELLOW}[!]${RESET} $*"; }
die()   { echo -e "${RED}[ошибка]${RESET} $*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1; }

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sw_chat_install.XXXXXX")"
cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT

echo -e "${CYAN}=== sw_chat installer ===${RESET}"
echo -e "${GRAY}репозиторий: ${GITHUB_USER}/${GITHUB_REPO}@${GITHUB_BRANCH}${RESET}"

detect_pm() {
  if need_cmd apt-get; then echo apt
  elif need_cmd dnf; then echo dnf
  elif need_cmd pacman; then echo pacman
  elif need_cmd zypper; then echo zypper
  else echo unknown
  fi
}

install_deps() {
  local pm="$1"
  if [[ "$SKIP_DEPS" -eq 1 ]]; then
    warn "пропуск установки зависимостей (--skip-deps)"
    need_cmd g++  || die "нужен g++"
    need_cmd curl || die "нужен curl"
    return 0
  fi
  info "зависимости (g++, readline, curl)…"
  case "$pm" in
    apt)
      if need_cmd sudo; then
        sudo apt-get update -qq
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
          g++ make curl ca-certificates \
          libreadline-dev libcurl4-openssl-dev
      else
        warn "sudo нет — проверяю уже установленное"
        need_cmd g++  || die "установите: g++ libreadline-dev libcurl4-openssl-dev curl"
        need_cmd curl || die "установите curl"
      fi
      ;;
    dnf)
      if need_cmd sudo; then
        sudo dnf install -y gcc-c++ make curl ca-certificates readline-devel libcurl-devel
      else
        need_cmd g++ || die "установите gcc-c++ readline-devel libcurl-devel curl"
      fi
      ;;
    pacman)
      if need_cmd sudo; then
        sudo pacman -Sy --noconfirm base-devel curl readline
      else
        need_cmd g++ || die "установите base-devel curl"
      fi
      ;;
    zypper)
      if need_cmd sudo; then
        sudo zypper install -y gcc-c++ make curl readline-devel libcurl-devel
      else
        need_cmd g++ || die "установите gcc-c++ readline-devel libcurl-devel curl"
      fi
      ;;
    *)
      warn "неизвестный пакетный менеджер"
      need_cmd g++  || die "нужен g++ (C++17)"
      need_cmd curl || die "нужен curl"
      warn "нужны dev-пакеты readline и libcurl"
      ;;
  esac
  ok "зависимости готовы"
}

install_nlohmann() {
  mkdir -p "$NLOHMANN_DIR"
  if [[ -f "$NLOHMANN_DIR/json.hpp" ]] && [[ "$(wc -c < "$NLOHMANN_DIR/json.hpp")" -gt 100000 ]]; then
    ok "nlohmann/json уже есть: $NLOHMANN_DIR/json.hpp"
    return 0
  fi
  info "скачиваю nlohmann/json…"
  curl -fsSL "$NLOHMANN_URL" -o "$NLOHMANN_DIR/json.hpp" || die "не удалось скачать nlohmann/json"
  [[ "$(wc -c < "$NLOHMANN_DIR/json.hpp")" -gt 100000 ]] || die "json.hpp слишком маленький"
  ok "nlohmann/json → $NLOHMANN_DIR/json.hpp"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"

download_source() {
  local dest="$TMP_DIR/sw_chat.cpp"
  if [[ "$USE_LOCAL" -eq 1 ]]; then
    local cand=""
    for c in "${SCRIPT_DIR}/sw_chat.cpp" "./sw_chat.cpp" "${HOME}/tmp/sw_chat.cpp"; do
      if [[ -f "$c" ]]; then cand="$c"; break; fi
    done
    [[ -n "$cand" ]] || die "--local: не найден sw_chat.cpp"
    info "локальный исходник: $cand"
    cp "$cand" "$dest"
  else
    info "скачиваю sw_chat.cpp с GitHub…"
    curl -fsSL "${REPO_RAW}/sw_chat.cpp" -o "$dest" || die "не удалось скачать ${REPO_RAW}/sw_chat.cpp"
  fi
  local sz; sz="$(wc -c < "$dest")"
  [[ "$sz" -gt 10000 ]] || die "исходник слишком маленький ($sz байт)"
  grep -q 'APP_VERSION' "$dest" || die "нет APP_VERSION — это не sw_chat.cpp?"
  local ver
  ver="$(grep -E '#define APP_VERSION' "$dest" | head -1 | sed -n 's/.*"\([^"]*\)".*/\1/p')"
  ok "исходник OK (${sz} байт, версия ${ver:-?})"
}

compile() {
  info "компиляция (C++17, -Os)…"
  g++ -std=c++17 -Os \
    -I"$INCLUDE_DIR" \
    -o "$TMP_DIR/sw_chat" \
    "$TMP_DIR/sw_chat.cpp" \
    -lreadline -lcurl -lpthread \
    || die "компиляция не удалась"
  [[ -x "$TMP_DIR/sw_chat" ]] || die "бинарник не создан"
  ok "сборка успешна"
}

install_binary() {
  info "установка в ${INSTALL_DIR}…"
  mkdir -p "$INSTALL_DIR"
  cp "$TMP_DIR/sw_chat" "$INSTALL_DIR/sw_chat"
  chmod +x "$INSTALL_DIR/sw_chat"
  ok "установлено: ${INSTALL_DIR}/sw_chat"
}

check_path() {
  case ":$PATH:" in
    *":${INSTALL_DIR}:"*) ok "${INSTALL_DIR} уже в PATH"; return 0 ;;
  esac
  warn "${INSTALL_DIR} нет в PATH"
  local shell_rc=""
  if [[ -f "${HOME}/.bashrc" ]]; then shell_rc="${HOME}/.bashrc"
  elif [[ -f "${HOME}/.zshrc" ]]; then shell_rc="${HOME}/.zshrc"
  fi
  local line='export PATH="$HOME/.local/bin:$PATH"'
  if [[ -n "$shell_rc" ]]; then
    if grep -qF '.local/bin' "$shell_rc" 2>/dev/null; then
      ok "PATH уже прописан в $shell_rc"
    else
      printf '\n# sw_chat\n%s\n' "$line" >> "$shell_rc"
      ok "добавлено в $shell_rc"
      warn "выполните: source $shell_rc  или откройте новый терминал"
    fi
  else
    warn "добавьте вручную: $line"
  fi
}

setup_api_key() {
  # Имя env начинается с цифры — bash не принимает ${302_API_KEY}, читаем через printenv
  local env_key
  env_key="$(printenv 302_API_KEY 2>/dev/null || true)"
  if [[ -n "$env_key" ]]; then
    ok "ключ найден в переменной 302_API_KEY"
    return 0
  fi
  if [[ -f "$KEY_FILE" ]] && [[ -s "$KEY_FILE" ]]; then
    ok "ключ найден: $KEY_FILE"
    return 0
  fi
  warn "API-ключ 302.ai не найден"
  echo -e "  Получите ключ: ${CYAN}https://302.ai${RESET}"
  echo -e "  Файл: ${GRAY}${KEY_FILE}${RESET}  ·  env: ${GRAY}302_API_KEY${RESET}"
  if [[ ! -t 0 ]]; then
    warn "stdin не TTY (curl|bash) — добавьте ключ позже:"
    echo -e "  ${GRAY}echo \"sk-...\" > ${KEY_FILE} && chmod 600 ${KEY_FILE}${RESET}"
    return 0
  fi
  local apikey=""
  read -r -p "Вставьте API-ключ (Enter — пропустить): " apikey || true
  if [[ -n "$apikey" ]]; then
    mkdir -p "$CONFIG_DIR"
    printf '%s\n' "$apikey" > "$KEY_FILE"
    chmod 600 "$KEY_FILE"
    ok "ключ сохранён: $KEY_FILE"
  else
    warn "пропущено. Позже: echo \"sk-...\" > ${KEY_FILE} && chmod 600 ${KEY_FILE}"
  fi
}

PM="$(detect_pm)"
info "пакетный менеджер: $PM"
install_deps "$PM"
install_nlohmann
download_source
compile
install_binary
check_path
setup_api_key

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════╗${RESET}"
echo -e "${GREEN}║     sw_chat успешно установлен             ║${RESET}"
echo -e "${GREEN}║  Запуск:  sw_chat                          ║${RESET}"
echo -e "${GREEN}║  Справка: /help  внутри программы          ║${RESET}"
echo -e "${GREEN}║  Ключ:    ~/.config/302_key                ║${RESET}"
echo -e "${GREEN}╚════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "${GRAY}Если sw_chat не находится:${RESET}"
echo -e "  ${INSTALL_DIR}/sw_chat"
echo -e "  export PATH=\"\$HOME/.local/bin:\$PATH\""
