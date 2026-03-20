# sw_chat — CLI-клиент для OpenRouter AI

Терминальный чат с языковыми моделями (Claude, GPT, Grok, Gemini и др.) через [OpenRouter](https://openrouter.ai).

## Возможности

- Красивый рендеринг Markdown в терминале (заголовки, таблицы, код, списки)
- Выполнение bash-блоков из ответов модели прямо в системе
- Многострочный ввод
- История диалога с автосохранением
- Переключение моделей на лету (`/model`)
- Подсчёт токенов и стоимости (`/cost`)
- Прерывание ответа по Ctrl+C
- Режим пайпа: `echo "вопрос" | sw_chat`

## Установка

### Одна команда:

    curl -fsSL https://raw.githubusercontent.com/swarik/Chat-Assist/main/install.sh | bash

### Вручную (Ubuntu/Debian):

    sudo apt-get install -y g++ libreadline-dev libcurl4-openssl-dev
    mkdir -p ~/.local/include/nlohmann
    curl -fsSL https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp \
         -o ~/.local/include/nlohmann/json.hpp
    curl -fsSL https://raw.githubusercontent.com/swarik/Chat-Assist/main/sw_chat.cpp > sw_chat.cpp
    g++ -std=c++17 -O2 -I~/.local/include -o sw_chat sw_chat.cpp -lreadline -lcurl
    cp sw_chat ~/.local/bin/

## API ключ

Получите ключ на [openrouter.ai](https://openrouter.ai) и сохраните:

    echo "sk-or-ваш-ключ" > ~/.config/openrouter_key
    chmod 600 ~/.config/openrouter_key

Или через переменную окружения:

    export OPENROUTER_API_KEY="sk-or-ваш-ключ"

## Использование

    sw_chat                         # интерактивный режим
    sw_chat "объясни что такое DNS" # одиночный вопрос
    echo "текст" | sw_chat          # режим пайпа

## Команды

| Команда           | Описание                        |
|-------------------|---------------------------------|
| `/help`           | справка                         |
| `/model`          | выбор модели                    |
| `/clear`          | очистить историю                |
| `/save` / `/load` | сохранить / загрузить историю   |
| `/file <path>`    | загрузить файл в контекст       |
| `/autorun`        | вкл/выкл авто-запуск bash       |
| `/temp [0-2]`     | температура модели              |
| `/cost`           | стоимость сессии в $            |
| `/retry`          | повторить последний запрос      |
| `/exit`           | выход                           |

## Зависимости

- `g++` с поддержкой C++17
- `libreadline`
- `libcurl`
- `nlohmann/json` (устанавливается автоматически скриптом)

## Поддерживаемые дистрибутивы

- Ubuntu / Debian
- Fedora / RHEL
- Arch Linux
- openSUSE
