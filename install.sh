#!/usr/bin/env bash
# sw_chat installer — Linux + Termux (Android)
#
#   curl -fsSL https://raw.githubusercontent.com/swarik/Chat-Assist/main/install.sh | bash
#   bash install.sh
#   bash install.sh --local
#   bash install.sh --termux     # принудительно режим Termux
#   bash install.sh --skip-deps
#
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

# nlohmann/json pin (не develop)
NLOHMANN_URL="https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp"

USE_LOCAL=0
SKIP_DEPS=0
FORCE_TERMUX=0
for arg in "$@"; do
  case "$arg" in
    --local|-l) USE_LOCAL=1 ;;
    --skip-deps) SKIP_DEPS=1 ;;
    --termux) FORCE_TERMUX=1 ;;
    -h|--help)
      cat <<'EOF'
Установка sw_chat (Linux / Termux)

  bash install.sh              # скачать с GitHub, собрать, поставить
  bash install.sh --local      # ./sw_chat.cpp рядом со скриптом
  bash install.sh --skip-deps  # не ставить пакеты
  bash install.sh --termux     # режим Termux (Android)

Linux:  ~/.local/bin/sw_chat
Termux: $PREFIX/bin/sw_chat  (обычно уже в PATH)

Ключ: ~/.config/302_key  или  env 302_API_KEY
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

# ── Termux? ───────────────────────────────────────────────────
is_termux() {
  if [[ "$FORCE_TERMUX" -eq 1 ]]; then return 0; fi
  # PREFIX + pkg — типичный Termux
  if [[ -n "${PREFIX:-}" && -d "${PREFIX}/bin" && -x "${PREFIX}/bin/pkg" ]]; then
    return 0
  fi
  if need_cmd termux-info; then return 0; fi
  if [[ "$(uname -o 2>/dev/null || true)" == "Android" ]]; then return 0; fi
  # legacy path
  if [[ -d /data/data/com.termux/files/usr ]]; then return 0; fi
  return 1
}

TERMUX=0
if is_termux; then TERMUX=1; fi

# Пути: на Termux ставим в $PREFIX (уже в PATH, без костылей)
if [[ "$TERMUX" -eq 1 ]]; then
  # на всякий случай, если PREFIX не экспортирован
  if [[ -z "${PREFIX:-}" ]]; then
    if [[ -d /data/data/com.termux/files/usr ]]; then
      PREFIX="/data/data/com.termux/files/usr"
      export PREFIX
    fi
  fi
  INSTALL_DIR="${PREFIX}/bin"
  INCLUDE_DIR="${PREFIX}/include"
  # config/key — в $HOME (Termux home = /data/data/com.termux/files/home)
  CONFIG_DIR="${HOME}/.config"
else
  INSTALL_DIR="${HOME}/.local/bin"
  INCLUDE_DIR="${HOME}/.local/include"
  CONFIG_DIR="${HOME}/.config"
fi
NLOHMANN_DIR="${INCLUDE_DIR}/nlohmann"
KEY_FILE="${CONFIG_DIR}/302_key"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sw_chat_install.XXXXXX")"
cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT

echo -e "${CYAN}=== sw_chat installer ===${RESET}"
echo -e "${GRAY}репозиторий: ${GITHUB_USER}/${GITHUB_REPO}@${GITHUB_BRANCH}${RESET}"
if [[ "$TERMUX" -eq 1 ]]; then
  echo -e "${GRAY}платформа: Termux (Android)  PREFIX=${PREFIX:-?}${RESET}"
else
  echo -e "${GRAY}платформа: Linux desktop/server${RESET}"
fi

# ── Компилятор ────────────────────────────────────────────────
pick_cxx() {
  # Termux: обычно clang++; g++ часто symlink на clang++
  if need_cmd clang++; then echo clang++
  elif need_cmd g++; then echo g++
  elif need_cmd c++; then echo c++
  else echo ""
  fi
}

# ── Пакетный менеджер ─────────────────────────────────────────
detect_pm() {
  if [[ "$TERMUX" -eq 1 ]]; then
    if need_cmd pkg; then echo termux
    elif need_cmd apt-get; then echo termux-apt
    else echo unknown
    fi
    return 0
  fi
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
    local cxx; cxx="$(pick_cxx)"
    [[ -n "$cxx" ]] || die "нужен clang++ или g++"
    need_cmd curl || die "нужен curl"
    return 0
  fi

  info "зависимости (компилятор, readline, curl)…"
  case "$pm" in
    termux|termux-apt)
      # Termux: root/sudo НЕ нужны
      # clang даёт clang++; readline/libcurl — dev-заголовки в тех же пакетах
      local tcmd=pkg
      need_cmd pkg || tcmd=apt
      info "Termux: $tcmd install … (без sudo)"
      # обновление индексов (мягко)
      if [[ "$tcmd" == "pkg" ]]; then
        pkg update -y || warn "pkg update не удался — пробуем install"
        pkg install -y clang make curl libcurl readline openssl ca-certificates \
          || die "pkg install не удался. Попробуйте: pkg install clang libcurl readline curl"
      else
        apt update -y || true
        apt install -y clang make curl libcurl readline openssl ca-certificates \
          || die "apt install в Termux не удался"
      fi
      # опционально: готовый nlohmann-json из репо Termux
      if [[ "$tcmd" == "pkg" ]]; then
        pkg install -y nlohmann-json 2>/dev/null || true
      fi
      ;;
    apt)
      if need_cmd sudo; then
        sudo apt-get update -qq
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
          g++ make curl ca-certificates \
          libreadline-dev libcurl4-openssl-dev
      else
        warn "sudo нет — проверяю уже установленное"
        need_cmd g++ || need_cmd clang++ || die "установите: g++ libreadline-dev libcurl4-openssl-dev curl"
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
      local cxx; cxx="$(pick_cxx)"
      [[ -n "$cxx" ]] || die "нужен clang++/g++ (C++17)"
      need_cmd curl || die "нужен curl"
      warn "нужны dev-пакеты readline и libcurl"
      ;;
  esac
  ok "зависимости готовы"
}

install_nlohmann() {
  mkdir -p "$NLOHMANN_DIR"
  # уже есть в PREFIX (Termux pkg) или в ~/.local
  local candidates=(
    "${NLOHMANN_DIR}/json.hpp"
    "${PREFIX:-/usr}/include/nlohmann/json.hpp"
    "/usr/include/nlohmann/json.hpp"
  )
  local f
  for f in "${candidates[@]}"; do
    if [[ -f "$f" ]] && [[ "$(wc -c < "$f")" -gt 100000 ]]; then
      ok "nlohmann/json: $f"
      # если нашли системный — добавим его include-корень позже через -I
      NLOHMANN_FOUND_DIR="$(dirname "$f")"
      return 0
    fi
  done
  info "скачиваю nlohmann/json…"
  curl -fsSL "$NLOHMANN_URL" -o "$NLOHMANN_DIR/json.hpp" || die "не удалось скачать nlohmann/json"
  [[ "$(wc -c < "$NLOHMANN_DIR/json.hpp")" -gt 100000 ]] || die "json.hpp слишком маленький"
  NLOHMANN_FOUND_DIR="$NLOHMANN_DIR"
  ok "nlohmann/json → $NLOHMANN_DIR/json.hpp"
}
NLOHMANN_FOUND_DIR="$NLOHMANN_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"

download_source() {
  local dest="$TMP_DIR/sw_chat.cpp"
  if [[ "$USE_LOCAL" -eq 1 ]]; then
    local cand="" c
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
  local cxx
  cxx="$(pick_cxx)"
  [[ -n "$cxx" ]] || die "не найден clang++/g++"

  # include: ~/.local или $PREFIX + каталог nlohmann
  local inc_flags=("-I${INCLUDE_DIR}")
  if [[ -n "${PREFIX:-}" && -d "${PREFIX}/include" ]]; then
    inc_flags+=("-I${PREFIX}/include")
  fi
  # если json.hpp лежит как .../nlohmann/json.hpp — -I parent(nlohmann)
  if [[ -f "${NLOHMANN_FOUND_DIR}/json.hpp" ]]; then
    # parent of nlohmann/
    local parent; parent="$(dirname "$NLOHMANN_FOUND_DIR")"
    inc_flags+=("-I${parent}")
  fi

  # pthread: на Android/Bionic часто в libc, но -pthread безопаснее чем -lpthread
  local thr_flag="-pthread"
  # на очень старых тулчейнах -pthread может ругаться — fallback ниже

  info "компиляция ($cxx, C++17, -Os)…"
  set +e
  "$cxx" -std=c++17 -Os \
    "${inc_flags[@]}" \
    -o "$TMP_DIR/sw_chat" \
    "$TMP_DIR/sw_chat.cpp" \
    -lreadline -lcurl $thr_flag 2>"$TMP_DIR/build.err"
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    # fallback: -lpthread вместо -pthread
    "$cxx" -std=c++17 -Os \
      "${inc_flags[@]}" \
      -o "$TMP_DIR/sw_chat" \
      "$TMP_DIR/sw_chat.cpp" \
      -lreadline -lcurl -lpthread 2>"$TMP_DIR/build2.err"
    rc=$?
  fi
  set -e
  if [[ $rc -ne 0 ]]; then
    warn "лог компиляции:"
    cat "$TMP_DIR/build.err" 2>/dev/null || true
    cat "$TMP_DIR/build2.err" 2>/dev/null || true
    die "компиляция не удалась"
  fi
  [[ -x "$TMP_DIR/sw_chat" ]] || die "бинарник не создан"
  ok "сборка успешна ($cxx)"
}

install_binary() {
  info "установка в ${INSTALL_DIR}…"
  mkdir -p "$INSTALL_DIR"
  # Termux: иногда $PREFIX/bin не writable только если сломан PREFIX — проверим
  if [[ ! -w "$INSTALL_DIR" ]]; then
    die "нет записи в $INSTALL_DIR"
  fi
  cp "$TMP_DIR/sw_chat" "$INSTALL_DIR/sw_chat"
  chmod +x "$INSTALL_DIR/sw_chat"
  ok "установлено: ${INSTALL_DIR}/sw_chat"
}

check_path() {
  case ":$PATH:" in
    *":${INSTALL_DIR}:"*) ok "${INSTALL_DIR} уже в PATH"; return 0 ;;
  esac
  # Termux $PREFIX/bin почти всегда в PATH; если нет — редкий кейс
  if [[ "$TERMUX" -eq 1 ]]; then
    warn "${INSTALL_DIR} нет в PATH (необычно для Termux)"
    warn "добавьте: export PATH=\"${INSTALL_DIR}:\$PATH\""
    return 0
  fi
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
  # env с цифры в начале: только printenv
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
    echo -e "  ${GRAY}mkdir -p ${CONFIG_DIR} && echo \"sk-...\" > ${KEY_FILE} && chmod 600 ${KEY_FILE}${RESET}"
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

termux_notes() {
  [[ "$TERMUX" -eq 1 ]] || return 0
  echo ""
  echo -e "${CYAN}Termux — кратко:${RESET}"
  echo -e "  • запуск: ${GREEN}sw_chat${RESET}"
  echo -e "  • ключ:   ${GRAY}~/.config/302_key${RESET}"
  echo -e "  • для API нужен интернет (Wi‑Fi / mobile data)"
  echo -e "  • при ошибках SSL: ${GRAY}pkg install ca-certificates${RESET}"
  echo -e "  • storage (опционально): ${GRAY}termux-setup-storage${RESET}"
}

# ── main ───────────────────────────────────────────────────────
PM="$(detect_pm)"
info "пакетный менеджер: $PM"
install_deps "$PM"
install_nlohmann
download_source
compile
install_binary
check_path
setup_api_key
termux_notes

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════╗${RESET}"
echo -e "${GREEN}║     sw_chat успешно установлен             ║${RESET}"
echo -e "${GREEN}║  Запуск:  sw_chat                          ║${RESET}"
echo -e "${GREEN}║  Справка: /help  внутри программы          ║${RESET}"
echo -e "${GREEN}║  Ключ:    ~/.config/302_key                ║${RESET}"
if [[ "$TERMUX" -eq 1 ]]; then
echo -e "${GREEN}║  Платформа: Termux                         ║${RESET}"
fi
echo -e "${GREEN}╚════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "${GRAY}Бинарник: ${INSTALL_DIR}/sw_chat${RESET}"
