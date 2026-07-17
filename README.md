# sw_chat

Терминальный CLI-чат с LLM через [302.ai](https://302.ai)  
(или другой **OpenAI-compatible** endpoint).

| | |
|---|---|
| **Версия** | 1.0.39 |
| **Язык** | C++17 |
| **Исходник** | [`sw_chat.cpp`](./sw_chat.cpp) |
| **Репозиторий** | [swarik/Chat-Assist](https://github.com/swarik/Chat-Assist) |

---

## Возможности

- Markdown в терминале (заголовки, таблицы, код, списки, emoji-width)
- Выполнение команд из ответов модели: fenced-блоки `bash` / `sh` (подтверждение y/n/a)
- Цепочка команд до 7 шагов (`bash chain N/7`)
- Код выхода и явная пометка пустого вывода в результате
- Многострочный ввод, сессии, история, алиасы, поиск, экспорт (md/txt/json)
- Модели: кэш, API, фильтр, пагинация (25 на страницу, cap 400)
- Оценка контекста (`~tok`), `/cost` и `/cost live`
- Self-update `/update` с preview
- Pipe/args; флаг `--exec` для обработки bash-блоков вне интерактива
- Режимы: compact, autorun, nores
- Low-RAM: `/file` <= 200 KB, обрезка длинных сообщений

### Безопасность

Модель может предлагать shell-команды. По умолчанию **каждый** блок требует подтверждения.

- Не включайте `/autorun` без необходимости на машине с ключами и личными данными
- Ответ `a` подтверждает все блоки текущего пакета
- Таймаут выполнения ограничен (`timeout` + `CMD_TIMEOUT`)

---

## Установка

### Скрипт

```text
curl -fsSL https://raw.githubusercontent.com/swarik/Chat-Assist/main/install.sh | bash
```

### Вручную (Ubuntu / Debian)

```text
sudo apt-get install -y g++ libreadline-dev libcurl4-openssl-dev

mkdir -p ~/.local/include/nlohmann
curl -fsSL https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp \
  -o ~/.local/include/nlohmann/json.hpp

curl -fsSL https://raw.githubusercontent.com/swarik/Chat-Assist/main/sw_chat.cpp -o sw_chat.cpp

g++ -std=c++17 -Os -I"$HOME/.local/include" -o sw_chat sw_chat.cpp \
  -lreadline -lcurl -lpthread

cp sw_chat ~/.local/bin/
# или: cp sw_chat ~/sw_chat && chmod +x ~/sw_chat
```

На слабом CPU предпочтительнее `-Os`.

### Локальная сборка из ~/tmp

```text
g++ -std=c++17 -Os -I"$HOME/.local/include" \
  -o ~/sw_chat ~/tmp/sw_chat.cpp -lreadline -lcurl -lpthread
```

---

## API-ключ

```text
mkdir -p ~/.config
echo "sk-ВАШ-КЛЮЧ" > ~/.config/302_key
chmod 600 ~/.config/302_key
```

Или env:

```text
export 302_API_KEY="sk-ВАШ-КЛЮЧ"
```

### Смена API base

По умолчанию: `https://api.302.ai`

```text
/apibase
/apibase https://api.302.ai
```

Сохраняется в `~/.config/sw_chat/config.json` (поле `api_base`).

URL:

- `{api_base}/v1/chat/completions`
- `{api_base}/v1/models`

---

## Системный промпт

Файл при старте:

```text
~/tmp/system_prompt.txt
```

Если нет — встроенный промпт. В чате: `/system`.

---

## Запуск

```text
sw_chat                              # интерактив
sw_chat "что такое DNS?"             # args
echo "привет" | sw_chat              # pipe: только ответ
echo "покажи дату" | sw_chat --exec  # pipe + bash-блоки
sw_chat --exec "кратко: uname"
```

| Флаг | Назначение |
|------|------------|
| `--exec`, `-e` | Обработка ответа с выполнением fenced bash/sh |
| `--restore-session` | Служебный флаг после `/update` |

### Многострочный ввод

| Действие | Ввод |
|----------|------|
| Отправить | пустой Enter или `//` |
| Пустая строка в тексте | `.` |
| Прервать API | Ctrl+C |
| Выход | `/exit` или Ctrl+D |

### Команды из ответа модели

Fenced-блок с языком `bash` или `sh`:

1. показ кода;
2. вопрос `y` / `n` / `a` (если не autorun);
3. выполнение с таймаутом;
4. результат в историю (user).

Пример результата:

```text
[Результат выполнения команды]:
...
[exit: 0]
```

Пустой stdout:

```text
(empty output, exit 0)
```

Цепочка: до **7** шагов, UI: `[bash chain N/7]`.

---

## Команды чата

- Кратко: `/help`
- Полностью: `/help all`

| Команда | Описание |
|---------|----------|
| `/help [all]` | Справка |
| `/model [name\|N]` | Выбор модели |
| `/models [filter] [pN\|page N] [refresh\|cache]` | Список / фильтр / страницы |
| `/apibase [url]` | API base |
| `/temp [0.0–2.0]` | Температура |
| `/maxtokens [N]` | max_tokens |
| `/system` | System prompt |
| `/file <path> [вопрос]` | Файл в контекст (<= **200 KB**) |
| `/save` / `/load` | История |
| `/history [on\|off]` | Показ / автосохранение |
| `/clear` | Очистить диалог |
| `/delete N` | Удалить сообщение N |
| `/retry` | Повтор запроса |
| `/tokens` | Токены сессии |
| `/cost [live]` | Стоимость $; live — цены API |
| `/balance` | Ключ, provider, токены |
| `/autorun` | Bash без подтверждения (**опасно**) |
| `/nores` | Скрыть вывод bash в TTY |
| `/compact` | Тихий режим |
| `/new [name]` | Новая сессия |
| `/list` | Сессии |
| `/switch <name>` | Переключить сессию |
| `/alias k=v` | Алиасы |
| `/search <text>` | Поиск |
| `/export [md\|txt\|json] [file]` | Экспорт |
| `/update` | Обновление + preview |
| `/about` | Версия, пути, api_base |
| `/exit` | Выход |

Tab-completion (readline) по командам и моделям.

---

## Файлы и конфигурация

| Путь | Назначение |
|------|------------|
| `~/.config/302_key` | API-ключ (или `302_API_KEY`) |
| `~/.config/sw_chat/config.json` | model, temp, max_tokens, autorun, history, nores, compact, **api_base**, aliases |
| `~/.config/sw_chat/sessions/*.json` | Сессии |
| `~/.config/sw_chat/models.json` | Кэш моделей (+ pricing) |
| `~/.config/sw_chat/.readline_history` | История ввода |
| `~/tmp/system_prompt.txt` | System prompt |
| `~/tmp/memo.md` | Опциональная память агента |

---

## Weak / low-RAM

Проверено на **32-bit Ubuntu 18.04**, ~**235 MB RAM**, слабом CPU.

- Сборка: `-Os`
- `/file` <= 200000 байт
- Сообщения ~120 KB max
- Models: cap 400, page 25; `/models p2`, `/models deepseek`
- Не держите `/autorun` постоянно

---

## Обновление

```text
/update
```

GitHub raw → `APP_VERSION` → preview → compile → `exec --restore-session`.

---

## Зависимости

- `g++` (C++17)
- `libreadline`
- `libcurl`
- [nlohmann/json](https://github.com/nlohmann/json) (header-only)
- `timeout` (coreutils)

### Дистрибутивы

- Ubuntu / Debian
- Fedora / RHEL
- Arch Linux
- openSUSE

---

## Changelog

### 1.0.39

- empty output + exit code в результатах bash
- `/models`: filter, pages, cap; pricing в кэше
- multiline → oneline в readline history
- preview при `/update`
- `/help` / `/help all`
- индикатор bash chain
- `--exec` для pipe/args
- `/apibase`
- оценка `~tok`
- `/cost live`

### 1.0.38

- лимиты `/file` и user-сообщений
- фильтр `--restore-session` из args
- валидный `/export json`, safe JSON
- сброс token counters при смене сессии
- fence `bash`/`sh`
- warning при autorun

---

## Лицензия

См. файлы репозитория (если лицензия добавлена отдельно).
Проект: [swarik/Chat-Assist](https://github.com/swarik/Chat-Assist).
