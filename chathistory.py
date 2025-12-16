import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

class ChatHistory:
    """Класс для управления историей чата"""
    
    def __init__(self, history_dir: str = "chat_history", compress_after: int = 12):
        """
        Инициализация менеджера истории
        
        Args:
            history_dir: директория для сохранения истории
        """
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(exist_ok=True)
        self.messages: list = []
        self.session_file: Path = None
        self.system_prompt: str = None
        self.temperature: float | None = 0.7
        self.max_tokens: int | None = None
        self.compress_after = compress_after
        self._summarizer_client: Any = None
        self._summarizer_model: str = "qwen/qwen-2.5-72b-instruct"
        self._summarizer: Callable[[list], str] | None = None
        # Статистика токенов для текущей сессии
        self.session_tokens = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "request_count": 0
        }
    
    def create_session(self, session_name: str = None) -> str:
        """
        Создать новую сессию чата
        
        Args:
            session_name: имя сессии (если None, используется временная метка)
        
        Returns:
            путь до файла сессии
        """
        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.session_file = self.history_dir / f"{session_name}.json"
        self.messages = []
        self.temperature = self.temperature
        self.max_tokens = None
        # Сбрасываем статистику токенов для новой сессии
        self.session_tokens = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "request_count": 0
        }
        self._save_session()
        
        print(f"✓ Новая сессия создана: {session_name}")
        return str(self.session_file)
    
    def load_session(self, session_name: str) -> bool:
        """
        Загрузить существующую сессию
        
        Args:
            session_name: имя сессии или путь до файла
        
        Returns:
            True если успешно, False если файл не найден
        """
        session_path = self.history_dir / f"{session_name}.json"
        
        if not session_path.exists():
            print(f"✗ Сессия не найдена: {session_name}")
            return False
        
        try:
            with open(session_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.messages = data.get('messages', [])
                self.system_prompt = data.get('system_prompt')
                self.temperature = data.get('temperature', self.temperature)
                self.max_tokens = data.get('max_tokens')
                self.compress_after = data.get('compress_after', self.compress_after)
                # Загружаем статистику токенов из сохраненной сессии
                self.session_tokens = data.get('session_tokens', {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "request_count": 0
                })
                self.session_file = session_path
                print(f"✓ Сессия загружена: {session_name} ({len(self.messages)} сообщений)")
                self._show_token_stats()
                return True
        except json.JSONDecodeError:
            print(f"✗ Ошибка при чтении файла сессии")
            return False
    
    def list_sessions(self) -> list:
        """Получить список всех сессий"""
        sessions = [f.stem for f in self.history_dir.glob("*.json")]
        return sorted(sessions, reverse=True)
    
    def add_message(self, role: str, content: str, metadata: dict = None) -> None:
        """
        Добавить сообщение в историю
        
        Args:
            role: роль (user, assistant, system)
            content: содержимое сообщения
            metadata: дополнительные данные (например, JSON-ответ)
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            message["metadata"] = metadata
        
        self.messages.append(message)
        self._maybe_compress()
        self._save_session()
    
    def get_messages_for_api(self) -> list:
        """
        Получить сообщения в формате для API (без служебных полей)
        
        Returns:
            список сообщений для отправки в API
        """
        result: list[dict] = []

        # Системный промпт пользователя всегда первым
        if self.system_prompt:
            result.append({"role": "system", "content": self.system_prompt})

        # Далее — сохранённые сообщения (включая сводки)
        for msg in self.messages:
            result.append(
                {
                    "role": msg["role"],
                    "content": msg["content"]
                }
            )
        return result
    
    def _save_session(self) -> None:
        """Сохранить текущую сессию в файл"""
        if self.session_file is None:
            return
        
        data = {
            "created": datetime.now().isoformat(),
            "message_count": len(self.messages),
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "compress_after": self.compress_after,
            "session_tokens": self.session_tokens,
            "messages": self.messages
        }
        
        with open(self.session_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def clear_history(self) -> None:
        """Очистить текущую историю в памяти"""
        self.messages = []
        if self.session_file:
            self._save_session()
    
    def show_history(self, limit: int = None) -> None:
        """
        Показать историю чата
        
        Args:
            limit: максимальное количество последних сообщений (None = все)
        """
        messages_to_show = self.messages[-limit:] if limit else self.messages
        
        print("\n" + "="*70)
        print("ИСТОРИЯ ЧАТА")
        print("="*70)
        
        if not messages_to_show:
            print("История пуста")
            return
        
        for i, msg in enumerate(messages_to_show, 1):
            role = msg["role"].upper()
            timestamp = msg["timestamp"]
            content = msg["content"]
            
            print(f"\n[{i}] {role} ({timestamp})")
            print(f"{'─'*70}")
            
            # Если есть метаданные (JSON), показываем их красиво
            if "metadata" in msg:
                print(f"Содержимое: {content}")
                print(f"JSON: {json.dumps(msg['metadata'], ensure_ascii=False, indent=2)}")
            else:
                print(f"{content}")
        
        print("\n" + "="*70 + "\n")
    
    def set_temperature(self, temperature: float | None) -> None:
        """Установить температуру и сохранить сессию"""
        self.temperature = temperature
        self._save_session()
    
    def export_history(self, filename: str = None) -> str:
        """
        Экспортировать историю в текстовый файл
        
        Args:
            filename: имя файла для экспорта
        
        Returns:
            путь до экспортированного файла
        """
        if filename is None:
            filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        export_path = self.history_dir / filename
        
        with open(export_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("ЭКСПОРТ ИСТОРИИ ЧАТА\n")
            f.write("="*70 + "\n\n")
            
            for i, msg in enumerate(self.messages, 1):
                f.write(f"[{i}] {msg['role'].upper()} ({msg['timestamp']})\n")
                f.write(f"{'─'*70}\n")
                f.write(f"{msg['content']}\n")
                
                if "metadata" in msg:
                    f.write(f"\nJSON: {json.dumps(msg['metadata'], ensure_ascii=False, indent=2)}\n")
                
                f.write("\n")
        
        print(f"✓ История экспортирована в: {export_path}")
        return str(export_path)
    
    def update_token_stats(self, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
        """
        Обновить статистику токенов для текущей сессии
        
        Args:
            prompt_tokens: количество токенов в промпте
            completion_tokens: количество токенов в ответе
            total_tokens: общее количество токенов
        """
        self.session_tokens["prompt_tokens"] += prompt_tokens
        self.session_tokens["completion_tokens"] += completion_tokens
        self.session_tokens["total_tokens"] += total_tokens
        self.session_tokens["request_count"] += 1
        self._save_session()
    
    def get_token_stats(self) -> dict:
        """
        Получить статистику токенов для текущей сессии
        
        Returns:
            словарь со статистикой токенов
        """
        return self.session_tokens.copy()
    
    def _show_token_stats(self) -> None:
        """Показать статистику токенов текущей сессии"""
        stats = self.session_tokens
        if stats["request_count"] > 0:
            print(f"📊 Статистика токенов сессии:")
            print(f"   Запросов: {stats['request_count']}")
            print(f"   Промпт токенов: {stats['prompt_tokens']}")
            print(f"   Ответ токенов: {stats['completion_tokens']}")
            print(f"   Всего токенов: {stats['total_tokens']}")
            print()
    
    def show_token_stats(self) -> None:
        """Показать статистику токенов текущей сессии (публичный метод)"""
        print("\n" + "="*70)
        print("СТАТИСТИКА ТОКЕНОВ СЕССИИ")
        print("="*70)
        
        stats = self.session_tokens
        if stats["request_count"] == 0:
            print("Статистика пуста (нет запросов в этой сессии)")
        else:
            print(f"Количество запросов: {stats['request_count']}")
            print(f"Промпт токенов:      {stats['prompt_tokens']:,}")
            print(f"Ответ токенов:       {stats['completion_tokens']:,}")
            print(f"Всего токенов:       {stats['total_tokens']:,}")
            if stats["request_count"] > 0:
                avg_prompt = stats['prompt_tokens'] / stats['request_count']
                avg_completion = stats['completion_tokens'] / stats['request_count']
                avg_total = stats['total_tokens'] / stats['request_count']
                print(f"\nСредние значения:")
                print(f"  Промпт токенов:    {avg_prompt:.1f}")
                print(f"  Ответ токенов:      {avg_completion:.1f}")
                print(f"  Всего токенов:      {avg_total:.1f}")
        
        print("="*70 + "\n")

    def set_max_tokens(self, max_tokens: int | None) -> None:
        """Установить ограничение на токены ответа и сохранить сессию"""
        self.max_tokens = max_tokens
        self._save_session()

    def set_compress_after(self, message_limit: int) -> None:
        """Установить порог сообщений для сжатия истории"""
        if message_limit < 4:
            raise ValueError("Порог сжатия должен быть не меньше 4 сообщений")
        self.compress_after = message_limit
        self._save_session()

    def set_summarizer(
        self,
        client: Any,
        model: str | None = None,
        custom_summarizer: Callable[[list], str] | None = None
    ) -> None:
        """
        Установить клиент и модель для авто-сжатия истории

        Args:
            client: совместимый клиент OpenAI
            model: название модели
            custom_summarizer: функция, возвращающая текст сжатия
        """
        self._summarizer_client = client
        if model:
            self._summarizer_model = model
        self._summarizer = custom_summarizer

    def _build_summary_prompt(self, messages: list) -> list[dict]:
        """Собирает промпт для модели с учетом требований к сводке"""
        history_text = "\n".join(
            [f"{m['role']}: {m['content']}" for m in messages]
        )
        system_text = (
            "Ты помогаешь с разработкой. Сожми историю диалога в пределах 400 токенов. "
            "Поддерживай контекст так: "
            "1) В контексте держи системные инструкции + последние 12 сообщений диалога. "
            "2) Всё, что старше, сворачивай в краткую сводку не более 400 токенов. "
            "Каждая сводка должна содержать: цель пользователя / проекта; "
            "принятые решения и ключевые выводы; важные ограничения "
            "(версии, дедлайны, бюджеты, API-лимиты); открытые вопросы или TODO. "
            "Правила: не добавляй новых фактов. Сохраняй точные термины, версии, "
            "номера задач/тикетов, пути файлов и ключевые команды. "
            "Если есть риск потери важной детали — добавь её как короткую цитату. "
            "При каждом обновлении диалога проверяй длину контекста; если приближается лимит — перегенерируй сводку. "
            "Формат сводки: bullet list или JSON с полями: goals, decisions, constraints, open_questions, todos. "
            "Кратко, без воды. Не добавляй новых фактов."
        )
        user_text = (
            "Сожми историю диалога ниже, соблюдай формат и ограничения.\n\n"
            f"История:\n{history_text}"
        )
        return [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text}
        ]

    def _run_summarizer(self, messages: list) -> str:
        """Выполняет сжатие истории (с запасным планом при ошибках)"""
        try:
            if self._summarizer:
                return self._summarizer(messages)

            if not self._summarizer_client:
                raise RuntimeError("Клиент summarizer не настроен")

            prompt = self._build_summary_prompt(messages)
            response = self._summarizer_client.chat.completions.create(
                model=self._summarizer_model,
                messages=prompt,
                temperature=0.2,
                max_tokens=600,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            # Резервное сжатие без LLM, чтобы не терять историю
            trimmed = "\n".join(
                [f"- {m['role']}: {m['content'][:160]}" for m in messages]
            )
            print(f"⚠️ Сжатие через модель не удалось: {exc}")
            return (
                "goals: []\n"
                "decisions: []\n"
                "constraints: []\n"
                "open_questions: []\n"
                f"todos: [\"Сжатие без модели. Краткий обзор:\\n{trimmed}\"]"
            )

    def _maybe_compress(self) -> None:
        """Автоматически сжимает историю, если превышен порог сообщений"""
        if len(self.messages) <= self.compress_after:
            return
        # Берём всё, что старше последних N сообщений
        to_summarize = self.messages[:-self.compress_after]
        if not to_summarize:
            return
        # Если в хвосте уже только сводка — не дублируем
        if len(to_summarize) == 1 and to_summarize[0].get("metadata", {}).get("type") == "summary":
            return

        summary_text = self._run_summarizer(to_summarize)
        summary_message = {
            "role": "system",
            "content": summary_text,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "type": "summary",
                "compressed_count": len(to_summarize)
            }
        }

        # Сохраняем сводку + хвост последних сообщений
        tail = self.messages[-self.compress_after:]
        tail = self._normalize_tail(tail)
        self.messages = [summary_message] + tail
        print(f"ℹ️ История сжата: {len(to_summarize)} сообщений свернуто")

    def _normalize_tail(self, tail: list[dict]) -> list[dict]:
        """
        Приводит хвост истории к формату: начинается с user/tool,
        далее роли чередуются user/tool -> assistant.
        """
        # убираем ведущие assistant/system
        while tail and tail[0]["role"] not in ("user", "tool"):
            tail = tail[1:]
        if not tail:
            return []

        normalized = [tail[0]]
        for msg in tail[1:]:
            if msg["role"] == normalized[-1]["role"]:
                # пропускаем дублирующую роль
                continue
            normalized.append(msg)
        return normalized
