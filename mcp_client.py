import json
import requests
from typing import Dict, List, Any, Optional


class MCPClient:
    """Клиент для работы с MCP сервером через HTTP"""
    
    def __init__(self, base_url: str = "http://0.0.0.0:8000"):
        """
        Инициализация MCP клиента
        
        Args:
            base_url: базовый URL MCP сервера
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def _make_request(self, method: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Выполнить JSON-RPC запрос к MCP серверу
        
        Args:
            method: имя метода JSON-RPC
            params: параметры запроса
        
        Returns:
            ответ от сервера
        """
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method
        }
        
        if params:
            payload["params"] = params
        
        try:
            response = self.session.post(
                f"{self.base_url}/mcp",
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ошибка подключения к MCP серверу: {str(e)}")
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Получить список доступных tools от MCP сервера
        
        Returns:
            список доступных tools
        """
        # Пробуем несколько вариантов подключения
        methods_to_try = [
            self._try_jsonrpc_method,
            self._try_direct_endpoint,
            self._try_sse_endpoint
        ]
        
        last_error = None
        for method in methods_to_try:
            try:
                tools = method()
                if tools:
                    return tools
            except Exception as e:
                last_error = e
                continue
        
        if last_error:
            raise last_error
        return []
    
    def _try_jsonrpc_method(self) -> List[Dict[str, Any]]:
        """Попытка получить tools через JSON-RPC метод tools/list"""
        response = self._make_request("tools/list")
        
        # Обрабатываем ответ JSON-RPC
        if "result" in response:
            result = response["result"]
            # MCP может вернуть tools в разных форматах
            if isinstance(result, list):
                return result
            elif isinstance(result, dict):
                return result.get("tools", [])
            return []
        elif "error" in response:
            error = response["error"]
            raise Exception(f"MCP сервер вернул ошибку: {error.get('message', 'Неизвестная ошибка')}")
        else:
            # Если формат ответа нестандартный
            if isinstance(response, list):
                return response
            return response.get("tools", [])
    
    def _try_direct_endpoint(self) -> List[Dict[str, Any]]:
        """Попытка получить tools через прямой GET запрос"""
        endpoints = ["/tools", "/mcp/tools", "/api/tools"]
        for endpoint in endpoints:
            try:
                response = self.session.get(f"{self.base_url}{endpoint}", timeout=10)
                response.raise_for_status()
                data = response.json()
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    return data.get("tools", [])
            except:
                continue
        raise Exception("Не удалось получить tools через прямые эндпоинты")
    
    def _try_sse_endpoint(self) -> List[Dict[str, Any]]:
        """Попытка получить tools через SSE эндпоинт (для streamable HTTP)"""
        try:
            # Для streamable HTTP может использоваться SSE
            response = self.session.get(f"{self.base_url}/sse/tools", timeout=10, stream=True)
            response.raise_for_status()
            # Если это не SSE, пробуем как обычный JSON
            data = response.json()
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("tools", [])
        except:
            pass
        raise Exception("Не удалось получить tools через SSE эндпоинт")
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Получить информацию о конкретном tool
        
        Args:
            tool_name: имя tool
        
        Returns:
            информация о tool или None если не найден
        """
        tools = self.list_tools()
        for tool in tools:
            if tool.get("name") == tool_name:
                return tool
        return None
    
    def test_connection(self) -> bool:
        """
        Проверить подключение к MCP серверу
        
        Returns:
            True если подключение успешно, False иначе
        """
        try:
            self.list_tools()
            return True
        except:
            return False

