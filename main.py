import os
import time
from openai import OpenAI
import json
from chathistory import ChatHistory
from mcp_client import MCPClient

# Получаем API ключ из переменных окружения
api_key = os.getenv("API_KEY")

if not api_key:
    print("Ошибка: установите переменную окружения API_KEY")
    exit(1)

# Инициализация клиента
client = OpenAI(
    api_key=api_key,
    base_url="https://api.perplexity.ai"
)

# System prompt для обработки запросов о погоде
SYSTEM_PROMPT = ""

def show_menu():
    """Показать меню команд"""
    print("\n" + "="*70)
    print("КОМАНДЫ:")
    print("="*70)
    print("  show      - показать историю")
    print("  sessions  - показать список сессий")
    print("  new       - создать новую сессию")
    print("  load      - загрузить сессию (введите имя)")
    print("  export    - экспортировать историю в текст")
    print("  clear     - очистить текущую историю")
    print("  temp      - установить температуру ответа")
    print("  max       - установить ограничение токенов ответа")
    print("  limit     - задать порог авто-сжатия истории")
    print("  tokens    - показать статистику токенов сессии")
    print("  change_system_prompt    - изменить system prompt")
    print("  mcp_tools - показать доступные tools от MCP сервера")
    print("  help      - показать это меню")
    print("  exit      - выход")
    print("="*70 + "\n")

def main():
    """Главная функция"""
    print("Консольный чат с Perplexity и сохранением истории")
    print("(введите 'help' для списка команд)\n")
    
    # Инициализируем менеджер истории
    history = ChatHistory()
    history.create_session()
    history.system_prompt = SYSTEM_PROMPT
    history.set_summarizer(client, model="sonar")
    
    # Инициализируем MCP клиент для gitflic
    mcp_client = MCPClient(base_url="http://0.0.0.0:8000")
    
    while True:
        try:
            user_input = input("Вы: ").strip()
            
            if not user_input:
                continue
            
            # Обработка команд
            if user_input.lower() == 'exit':
                print("До свидания!")
                break
            
            elif user_input.lower() == 'help':
                show_menu()
            
            elif user_input.lower() == 'show':
                history.show_history()
            
            elif user_input.lower() == 'sessions':
                sessions = history.list_sessions()
                print("\nДоступные сессии:")
                for i, session in enumerate(sessions, 1):
                    print(f"  {i}. {session}")
                print()
            
            elif user_input.lower() == 'new':
                name = input("Введите имя новой сессии (или Enter для автоматического): ").strip()
                history.create_session(name if name else None)
            
            elif user_input.lower() == 'load':
                name = input("Введите имя сессии для загрузки: ").strip()
                history.load_session(name)
            
            elif user_input.lower() == 'export':
                filename = input("Введите имя файла (или Enter для автоматического): ").strip()
                history.export_history(filename if filename else None)
            
            elif user_input.lower() == 'clear':
                confirm = input("Вы уверены? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    history.clear_history()
                    print("✓ История очищена\n")
            elif user_input.lower() == 'change_system_prompt':
                print("Введите новый system prompt\n")
                history.system_prompt = input("Вы: ").strip()
            elif user_input.lower() == 'temp':
                value_str = input("Задайте температуру (0-2): ").strip()
                if not value_str:
                    print("✗ Температура не изменена (пустой ввод)\n")
                    continue
                try:
                    value = float(value_str)
                    if value < 0 or value > 1.9:
                        raise ValueError
                    history.set_temperature(value)
                    print(f"✓ Температура установлена: {value}\n")
                except ValueError:
                    print("✗ Неверное значение температуры. Введите число от 0 до 2\n")
                continue
            
            elif user_input.lower() == 'max':
                value_str = input("Задайте максимум токенов ответа (число или пусто чтобы снять): ").strip()
                if not value_str:
                    history.set_max_tokens(None)
                    print("✓ Ограничение снято (без лимита токенов)\n")
                    continue
                try:
                    value = int(value_str)
                    if value <= 0:
                        raise ValueError
                    history.set_max_tokens(value)
                    print(f"✓ Максимум токенов ответа установлен: {value}\n")
                except ValueError:
                    print("✗ Неверное значение. Введите положительное целое число или оставьте пустым\n")
                continue
            
            elif user_input.lower() == 'tokens':
                history.show_token_stats()
                continue
            elif user_input.lower() == 'mcp_tools':
                print("\n" + "="*70)
                print("ДОСТУПНЫЕ TOOLS ОТ MCP СЕРВЕРА")
                print("="*70)
                try:
                    tools = mcp_client.list_tools()
                    if not tools:
                        print("Доступные tools не найдены или сервер не отвечает")
                    else:
                        for i, tool in enumerate(tools, 1):
                            name = tool.get("name", "Без имени")
                            description = tool.get("description", "Нет описания")
                            print(f"\n[{i}] {name}")
                            print(f"    Описание: {description}")
                            
                            # Выводим параметры если есть
                            if "inputSchema" in tool:
                                schema = tool["inputSchema"]
                                if "properties" in schema:
                                    print("    Параметры:")
                                    for param_name, param_info in schema["properties"].items():
                                        param_type = param_info.get("type", "unknown")
                                        param_desc = param_info.get("description", "")
                                        required = param_name in schema.get("required", [])
                                        req_mark = " (обязательный)" if required else ""
                                        print(f"      - {param_name} ({param_type}){req_mark}")
                                        if param_desc:
                                            print(f"        {param_desc}")
                    print("\n" + "="*70 + "\n")
                except Exception as e:
                    print(f"\n✗ Ошибка при получении tools: {str(e)}\n")
                    print("Проверьте, что MCP сервер запущен на http://0.0.0.0:8000\n")
                continue
            elif user_input.lower() == 'limit':
                value_str = input(f"После скольких сообщений сжимать историю? (текущее {history.compress_after}): ").strip()
                if not value_str:
                    print(f"Текущий порог: {history.compress_after} сообщений\n")
                    continue
                try:
                    value = int(value_str)
                    history.set_compress_after(value)
                    print(f"✓ Порог сжатия установлен: {value}\n")
                except ValueError:
                    print("✗ Неверное значение. Введите целое число не меньше 4\n")
                continue
            # Обработка запроса к API
            else:
                # Добавляем сообщение пользователя в историю
                history.add_message("user", user_input)
                
                # Формируем сообщения с system prompt
                # messages_for_api = [
                #     {
                #         "role": "system",
                #         "content": history.system_prompt or SYSTEM_PROMPT
                #     }
                # ] + history.get_messages_for_api()
                
                messages_for_api = history.get_messages_for_api()
                
                # Делаем запрос к API и замеряем время ответа
                model = "sonar-pro"
                started_at = time.perf_counter()
                response = client.chat.completions.create(
                    model=model,
                    messages=messages_for_api,
                    temperature=history.temperature,
                    max_tokens=history.max_tokens,
                    extra_body={
                        "disable_search": True
                    }
                )
                elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
                total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) if usage else prompt_tokens + completion_tokens
                metrics = {
                    "response_time_ms": elapsed_ms,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
                
                # Обновляем статистику токенов для сессии
                history.update_token_stats(prompt_tokens, completion_tokens, total_tokens)
                
                response_text = response.choices[0].message.content
                
                # Пытаемся парсить JSON
                try:
                    weather_data = json.loads(response_text)
                    formatted_response = json.dumps(weather_data, ensure_ascii=False, indent=2)
                    
                    # Сохраняем с метаданными
                    metadata = {
                        "data": weather_data,
                        "response_metrics": metrics
                    }
                    history.add_message(
                        "assistant",
                        formatted_response,
                        metadata=metadata
                    )
                    print(f"\nМетрики запроса: {metrics}")
                    print(f"Статистика сессии: {history.get_token_stats()}\n")
                    print(f"\nПомощник:\n{formatted_response}\n")
                
                except json.JSONDecodeError:
                    # Если не удалось парсить JSON, сохраняем как текст
                    history.add_message(
                        "assistant",
                        response_text,
                        metadata={"response_metrics": metrics}
                    )
                    print(f"\nМетрики запроса: {metrics}")
                    print(f"Статистика сессии: {history.get_token_stats()}\n")
                    print(f"\nПомощник: {response_text}\n")
        
        except KeyboardInterrupt:
            print("\n\nПрограмма прервана пользователем")
            break
        except Exception as e:
            print(f"\n✗ Ошибка: {str(e)}\n")


if __name__ == "__main__":
    main()