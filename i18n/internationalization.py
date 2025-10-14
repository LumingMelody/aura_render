"""
Internationalization (i18n) System

Comprehensive multi-language support system with dynamic translation loading,
language detection, and locale-specific formatting.
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import logging

from fastapi import Request, Depends
from babel import Locale, dates, numbers, core
from babel.messages import Catalog
from babel.messages.pofile import read_po, write_po
import gettext

logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Supported languages"""
    ENGLISH = "en"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    JAPANESE = "ja"
    KOREAN = "ko"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    ARABIC = "ar"
    HINDI = "hi"


@dataclass
class LanguageInfo:
    """Language information"""
    code: str
    name: str
    native_name: str
    rtl: bool = False
    locale: str = None
    
    def __post_init__(self):
        if self.locale is None:
            self.locale = self.code


class TranslationKey:
    """Translation key constants"""
    # Common
    WELCOME = "common.welcome"
    HELLO = "common.hello"
    GOODBYE = "common.goodbye"
    THANK_YOU = "common.thank_you"
    PLEASE = "common.please"
    YES = "common.yes"
    NO = "common.no"
    OK = "common.ok"
    CANCEL = "common.cancel"
    SAVE = "common.save"
    DELETE = "common.delete"
    EDIT = "common.edit"
    CREATE = "common.create"
    UPDATE = "common.update"
    LOADING = "common.loading"
    ERROR = "common.error"
    SUCCESS = "common.success"
    WARNING = "common.warning"
    INFO = "common.info"
    
    # Authentication
    LOGIN = "auth.login"
    LOGOUT = "auth.logout"
    REGISTER = "auth.register"
    EMAIL = "auth.email"
    PASSWORD = "auth.password"
    FORGOT_PASSWORD = "auth.forgot_password"
    INVALID_CREDENTIALS = "auth.invalid_credentials"
    ACCOUNT_LOCKED = "auth.account_locked"
    
    # Video Generation
    VIDEO_GENERATION = "video.generation"
    THEME = "video.theme"
    KEYWORDS = "video.keywords"
    DURATION = "video.duration"
    QUALITY = "video.quality"
    GENERATE = "video.generate"
    GENERATING = "video.generating"
    GENERATED = "video.generated"
    PREVIEW = "video.preview"
    DOWNLOAD = "video.download"
    
    # AI Optimization
    AI_OPTIMIZATION = "ai.optimization"
    OPTIMIZE = "ai.optimize"
    OPTIMIZING = "ai.optimizing"
    OPTIMIZATION_LEVEL = "ai.optimization_level"
    QUALITY_ENHANCEMENT = "ai.quality_enhancement"
    
    # Errors
    ERROR_GENERIC = "errors.generic"
    ERROR_NOT_FOUND = "errors.not_found"
    ERROR_UNAUTHORIZED = "errors.unauthorized"
    ERROR_FORBIDDEN = "errors.forbidden"
    ERROR_VALIDATION = "errors.validation"
    ERROR_SERVER = "errors.server"
    ERROR_NETWORK = "errors.network"
    
    # Admin
    ADMIN_PANEL = "admin.panel"
    USER_MANAGEMENT = "admin.user_management"
    SYSTEM_SETTINGS = "admin.system_settings"
    BACKUP_RESTORE = "admin.backup_restore"
    ANALYTICS = "admin.analytics"


class I18nManager:
    """Main internationalization manager"""
    
    def __init__(self, base_dir: str = "i18n"):
        self.base_dir = Path(base_dir)
        self.default_language = SupportedLanguage.ENGLISH.value
        self.translations: Dict[str, Dict[str, str]] = {}
        self.language_info: Dict[str, LanguageInfo] = {}
        self.current_language = self.default_language
        self.fallback_chain = [self.default_language]
        
        # Initialize language information
        self._initialize_language_info()
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _initialize_language_info(self):
        """Initialize supported language information"""
        self.language_info = {
            SupportedLanguage.ENGLISH.value: LanguageInfo(
                code="en",
                name="English",
                native_name="English",
                locale="en_US"
            ),
            SupportedLanguage.CHINESE_SIMPLIFIED.value: LanguageInfo(
                code="zh-CN",
                name="Chinese (Simplified)",
                native_name="简体中文",
                locale="zh_CN"
            ),
            SupportedLanguage.CHINESE_TRADITIONAL.value: LanguageInfo(
                code="zh-TW",
                name="Chinese (Traditional)",
                native_name="繁體中文",
                locale="zh_TW"
            ),
            SupportedLanguage.JAPANESE.value: LanguageInfo(
                code="ja",
                name="Japanese",
                native_name="日本語",
                locale="ja_JP"
            ),
            SupportedLanguage.KOREAN.value: LanguageInfo(
                code="ko",
                name="Korean",
                native_name="한국어",
                locale="ko_KR"
            ),
            SupportedLanguage.SPANISH.value: LanguageInfo(
                code="es",
                name="Spanish",
                native_name="Español",
                locale="es_ES"
            ),
            SupportedLanguage.FRENCH.value: LanguageInfo(
                code="fr",
                name="French",
                native_name="Français",
                locale="fr_FR"
            ),
            SupportedLanguage.GERMAN.value: LanguageInfo(
                code="de",
                name="German",
                native_name="Deutsch",
                locale="de_DE"
            ),
            SupportedLanguage.ITALIAN.value: LanguageInfo(
                code="it",
                name="Italian",
                native_name="Italiano",
                locale="it_IT"
            ),
            SupportedLanguage.PORTUGUESE.value: LanguageInfo(
                code="pt",
                name="Portuguese",
                native_name="Português",
                locale="pt_PT"
            ),
            SupportedLanguage.RUSSIAN.value: LanguageInfo(
                code="ru",
                name="Russian",
                native_name="Русский",
                locale="ru_RU"
            ),
            SupportedLanguage.ARABIC.value: LanguageInfo(
                code="ar",
                name="Arabic",
                native_name="العربية",
                rtl=True,
                locale="ar_SA"
            ),
            SupportedLanguage.HINDI.value: LanguageInfo(
                code="hi",
                name="Hindi",
                native_name="हिन्दी",
                locale="hi_IN"
            )
        }
    
    def _ensure_directories(self):
        """Ensure translation directories exist"""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        for lang_code in self.language_info.keys():
            lang_dir = self.base_dir / lang_code
            lang_dir.mkdir(exist_ok=True)
    
    async def initialize(self):
        """Initialize the i18n system"""
        # Load default English translations
        await self._create_default_translations()
        
        # Load all available translations
        await self._load_all_translations()
        
        logger.info(f"I18n system initialized with {len(self.translations)} languages")
    
    async def _create_default_translations(self):
        """Create default English translations if they don't exist"""
        default_translations = {
            # Common translations
            TranslationKey.WELCOME: "Welcome to Aura Render",
            TranslationKey.HELLO: "Hello",
            TranslationKey.GOODBYE: "Goodbye",
            TranslationKey.THANK_YOU: "Thank you",
            TranslationKey.PLEASE: "Please",
            TranslationKey.YES: "Yes",
            TranslationKey.NO: "No",
            TranslationKey.OK: "OK",
            TranslationKey.CANCEL: "Cancel",
            TranslationKey.SAVE: "Save",
            TranslationKey.DELETE: "Delete",
            TranslationKey.EDIT: "Edit",
            TranslationKey.CREATE: "Create",
            TranslationKey.UPDATE: "Update",
            TranslationKey.LOADING: "Loading...",
            TranslationKey.ERROR: "Error",
            TranslationKey.SUCCESS: "Success",
            TranslationKey.WARNING: "Warning",
            TranslationKey.INFO: "Information",
            
            # Authentication
            TranslationKey.LOGIN: "Login",
            TranslationKey.LOGOUT: "Logout",
            TranslationKey.REGISTER: "Register",
            TranslationKey.EMAIL: "Email",
            TranslationKey.PASSWORD: "Password",
            TranslationKey.FORGOT_PASSWORD: "Forgot Password",
            TranslationKey.INVALID_CREDENTIALS: "Invalid email or password",
            TranslationKey.ACCOUNT_LOCKED: "Account is temporarily locked",
            
            # Video Generation
            TranslationKey.VIDEO_GENERATION: "Video Generation",
            TranslationKey.THEME: "Theme",
            TranslationKey.KEYWORDS: "Keywords",
            TranslationKey.DURATION: "Duration",
            TranslationKey.QUALITY: "Quality",
            TranslationKey.GENERATE: "Generate",
            TranslationKey.GENERATING: "Generating...",
            TranslationKey.GENERATED: "Generated",
            TranslationKey.PREVIEW: "Preview",
            TranslationKey.DOWNLOAD: "Download",
            
            # AI Optimization
            TranslationKey.AI_OPTIMIZATION: "AI Optimization",
            TranslationKey.OPTIMIZE: "Optimize",
            TranslationKey.OPTIMIZING: "Optimizing...",
            TranslationKey.OPTIMIZATION_LEVEL: "Optimization Level",
            TranslationKey.QUALITY_ENHANCEMENT: "Quality Enhancement",
            
            # Errors
            TranslationKey.ERROR_GENERIC: "An error occurred",
            TranslationKey.ERROR_NOT_FOUND: "Not found",
            TranslationKey.ERROR_UNAUTHORIZED: "Unauthorized",
            TranslationKey.ERROR_FORBIDDEN: "Access forbidden",
            TranslationKey.ERROR_VALIDATION: "Validation error",
            TranslationKey.ERROR_SERVER: "Server error",
            TranslationKey.ERROR_NETWORK: "Network error",
            
            # Admin
            TranslationKey.ADMIN_PANEL: "Admin Panel",
            TranslationKey.USER_MANAGEMENT: "User Management",
            TranslationKey.SYSTEM_SETTINGS: "System Settings",
            TranslationKey.BACKUP_RESTORE: "Backup & Restore",
            TranslationKey.ANALYTICS: "Analytics"
        }
        
        # Save English translations
        await self._save_translations(self.default_language, default_translations)
        
        # Create sample translations for Chinese
        chinese_translations = {
            TranslationKey.WELCOME: "欢迎使用 Aura Render",
            TranslationKey.HELLO: "你好",
            TranslationKey.GOODBYE: "再见",
            TranslationKey.THANK_YOU: "谢谢",
            TranslationKey.PLEASE: "请",
            TranslationKey.YES: "是",
            TranslationKey.NO: "否",
            TranslationKey.OK: "确定",
            TranslationKey.CANCEL: "取消",
            TranslationKey.SAVE: "保存",
            TranslationKey.DELETE: "删除",
            TranslationKey.EDIT: "编辑",
            TranslationKey.CREATE: "创建",
            TranslationKey.UPDATE: "更新",
            TranslationKey.LOADING: "加载中...",
            TranslationKey.ERROR: "错误",
            TranslationKey.SUCCESS: "成功",
            TranslationKey.WARNING: "警告",
            TranslationKey.INFO: "信息",
            TranslationKey.VIDEO_GENERATION: "视频生成",
            TranslationKey.THEME: "主题",
            TranslationKey.KEYWORDS: "关键词",
            TranslationKey.DURATION: "时长",
            TranslationKey.QUALITY: "质量",
            TranslationKey.GENERATE: "生成",
            TranslationKey.GENERATING: "生成中...",
            TranslationKey.AI_OPTIMIZATION: "AI 优化",
            TranslationKey.OPTIMIZE: "优化",
            TranslationKey.OPTIMIZING: "优化中...",
        }
        
        await self._save_translations("zh-CN", chinese_translations)
    
    async def _save_translations(self, language: str, translations: Dict[str, str]):
        """Save translations to file"""
        try:
            translation_file = self.base_dir / language / "messages.json"
            translation_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(translation_file, 'w', encoding='utf-8') as f:
                json.dump(translations, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(translations)} translations for {language}")
            
        except Exception as e:
            logger.error(f"Failed to save translations for {language}: {e}")
    
    async def _load_all_translations(self):
        """Load all available translations"""
        for lang_code in self.language_info.keys():
            await self._load_translations(lang_code)
    
    async def _load_translations(self, language: str):
        """Load translations for a specific language"""
        try:
            translation_file = self.base_dir / language / "messages.json"
            
            if translation_file.exists():
                with open(translation_file, 'r', encoding='utf-8') as f:
                    translations = json.load(f)
                
                self.translations[language] = translations
                logger.info(f"Loaded {len(translations)} translations for {language}")
            else:
                self.translations[language] = {}
                logger.warning(f"No translations found for {language}")
                
        except Exception as e:
            logger.error(f"Failed to load translations for {language}: {e}")
            self.translations[language] = {}
    
    def set_language(self, language: str):
        """Set current language"""
        if language in self.language_info:
            self.current_language = language
            logger.info(f"Language set to {language}")
        else:
            logger.warning(f"Unsupported language: {language}")
    
    def get_supported_languages(self) -> List[LanguageInfo]:
        """Get list of supported languages"""
        return list(self.language_info.values())
    
    def detect_language(self, request: Request) -> str:
        """Detect user's preferred language from request"""
        # Check URL parameter
        lang_param = request.query_params.get('lang')
        if lang_param and lang_param in self.language_info:
            return lang_param
        
        # Check headers
        accept_language = request.headers.get('Accept-Language')
        if accept_language:
            # Parse Accept-Language header
            languages = []
            for lang_range in accept_language.split(','):
                parts = lang_range.strip().split(';')
                lang = parts[0].strip()
                
                # Extract quality factor
                quality = 1.0
                if len(parts) > 1 and parts[1].strip().startswith('q='):
                    try:
                        quality = float(parts[1].strip()[2:])
                    except ValueError:
                        quality = 1.0
                
                languages.append((lang, quality))
            
            # Sort by quality and find best match
            languages.sort(key=lambda x: x[1], reverse=True)
            
            for lang, _ in languages:
                # Check exact match
                if lang in self.language_info:
                    return lang
                
                # Check language family match (e.g., zh for zh-CN)
                lang_family = lang.split('-')[0]
                for supported_lang in self.language_info.keys():
                    if supported_lang.startswith(lang_family):
                        return supported_lang
        
        # Return default language
        return self.default_language
    
    def translate(self, key: str, language: str = None, **kwargs) -> str:
        """Translate a key to the specified language"""
        if language is None:
            language = self.current_language
        
        # Try to get translation from specified language
        if language in self.translations and key in self.translations[language]:
            translation = self.translations[language][key]
        else:
            # Fall back to default language
            if self.default_language in self.translations and key in self.translations[self.default_language]:
                translation = self.translations[self.default_language][key]
            else:
                # Return key if no translation found
                translation = key
        
        # Format translation with provided arguments
        try:
            if kwargs:
                translation = translation.format(**kwargs)
        except (KeyError, ValueError) as e:
            logger.warning(f"Translation formatting error for key {key}: {e}")
        
        return translation
    
    def translate_many(self, keys: List[str], language: str = None) -> Dict[str, str]:
        """Translate multiple keys"""
        return {key: self.translate(key, language) for key in keys}
    
    def format_datetime(self, dt: datetime, language: str = None, format: str = 'medium') -> str:
        """Format datetime according to language locale"""
        if language is None:
            language = self.current_language
        
        lang_info = self.language_info.get(language)
        if not lang_info:
            lang_info = self.language_info[self.default_language]
        
        try:
            locale = Locale.parse(lang_info.locale)
            return dates.format_datetime(dt, format=format, locale=locale)
        except Exception as e:
            logger.warning(f"DateTime formatting error: {e}")
            return str(dt)
    
    def format_date(self, date, language: str = None, format: str = 'medium') -> str:
        """Format date according to language locale"""
        if language is None:
            language = self.current_language
        
        lang_info = self.language_info.get(language)
        if not lang_info:
            lang_info = self.language_info[self.default_language]
        
        try:
            locale = Locale.parse(lang_info.locale)
            return dates.format_date(date, format=format, locale=locale)
        except Exception as e:
            logger.warning(f"Date formatting error: {e}")
            return str(date)
    
    def format_currency(self, amount: float, currency: str = 'USD', language: str = None) -> str:
        """Format currency according to language locale"""
        if language is None:
            language = self.current_language
        
        lang_info = self.language_info.get(language)
        if not lang_info:
            lang_info = self.language_info[self.default_language]
        
        try:
            locale = Locale.parse(lang_info.locale)
            return numbers.format_currency(amount, currency, locale=locale)
        except Exception as e:
            logger.warning(f"Currency formatting error: {e}")
            return f"{amount} {currency}"
    
    def format_number(self, number: Union[int, float], language: str = None) -> str:
        """Format number according to language locale"""
        if language is None:
            language = self.current_language
        
        lang_info = self.language_info.get(language)
        if not lang_info:
            lang_info = self.language_info[self.default_language]
        
        try:
            locale = Locale.parse(lang_info.locale)
            if isinstance(number, float):
                return numbers.format_decimal(number, locale=locale)
            else:
                return numbers.format_number(number, locale=locale)
        except Exception as e:
            logger.warning(f"Number formatting error: {e}")
            return str(number)
    
    def is_rtl(self, language: str = None) -> bool:
        """Check if language is right-to-left"""
        if language is None:
            language = self.current_language
        
        lang_info = self.language_info.get(language)
        return lang_info.rtl if lang_info else False
    
    async def add_translation(self, language: str, key: str, value: str):
        """Add or update a translation"""
        if language not in self.translations:
            self.translations[language] = {}
        
        self.translations[language][key] = value
        
        # Save to file
        await self._save_translations(language, self.translations[language])
    
    async def remove_translation(self, language: str, key: str):
        """Remove a translation"""
        if language in self.translations and key in self.translations[language]:
            del self.translations[language][key]
            await self._save_translations(language, self.translations[language])
    
    def get_translation_stats(self) -> Dict[str, Any]:
        """Get translation statistics"""
        stats = {
            "total_languages": len(self.translations),
            "total_keys": len(self.translations.get(self.default_language, {})),
            "languages": {}
        }
        
        default_keys = set(self.translations.get(self.default_language, {}).keys())
        
        for lang, translations in self.translations.items():
            lang_keys = set(translations.keys())
            completion = len(lang_keys & default_keys) / len(default_keys) if default_keys else 0
            
            stats["languages"][lang] = {
                "total_translations": len(translations),
                "completion_percentage": completion * 100,
                "missing_keys": list(default_keys - lang_keys)
            }
        
        return stats


# Global i18n manager
i18n_manager: Optional[I18nManager] = None


async def initialize_i18n(base_dir: str = "i18n"):
    """Initialize global i18n system"""
    global i18n_manager
    i18n_manager = I18nManager(base_dir)
    await i18n_manager.initialize()


def get_i18n() -> I18nManager:
    """Get global i18n manager"""
    if i18n_manager is None:
        raise RuntimeError("I18n system not initialized")
    return i18n_manager


def get_user_language(request: Request) -> str:
    """Dependency to get user's preferred language"""
    if i18n_manager:
        return i18n_manager.detect_language(request)
    return SupportedLanguage.ENGLISH.value


def translate(key: str, language: str = None, **kwargs) -> str:
    """Global translate function"""
    if i18n_manager:
        return i18n_manager.translate(key, language, **kwargs)
    return key


class TranslationMiddleware:
    """Middleware to set language for each request"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request = Request(scope, receive)
            
            # Detect and set language
            if i18n_manager:
                language = i18n_manager.detect_language(request)
                i18n_manager.set_language(language)
                
                # Add language to request state
                request.state.language = language
        
        await self.app(scope, receive, send)