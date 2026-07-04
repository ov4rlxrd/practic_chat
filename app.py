import sys
import threading
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QLineEdit, QFileDialog,
    QScrollArea, QFrame, QSizePolicy, QSpacerItem, QComboBox
)
from PySide6.QtCore import Qt, Signal, QObject, QThread, QSize, Slot
from PySide6.QtGui import QFont, QColor, QPalette, QIcon, QTextCursor

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False



DARK_BG       = "#0f0f0f"
SIDEBAR_BG    = "#161616"
CHAT_BG       = "#111111"
BUBBLE_USER   = "#1e1e2e"
BUBBLE_AI     = "#191919"
ACCENT        = "#7c6af7"
ACCENT_HOVER  = "#9585f9"
TEXT_PRIMARY  = "#e8e8e8"
TEXT_MUTED    = "#555555"
BORDER        = "#252525"
INPUT_BG      = "#1a1a1a"

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {DARK_BG};
    color: {TEXT_PRIMARY};
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 14px;
}}

/* ── Боковая панель ── */
#sidebar {{
    background-color: {SIDEBAR_BG};
    border-right: 1px solid {BORDER};
    min-width: 220px;
    max-width: 220px;
}}

#appTitle {{
    font-size: 17px;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    padding: 24px 20px 8px 20px;
    letter-spacing: 0.5px;
}}

#modelLabel {{
    font-size: 11px;
    color: {TEXT_MUTED};
    padding: 16px 20px 6px 20px;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

#modelName {{
    font-size: 13px;
    color: {TEXT_PRIMARY};
    padding: 20px 20px;
    word-wrap: break-word;
}}

#loadBtn {{
    background-color: {ACCENT};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 13px;
    font-weight: 600;
    margin: 16px;
    text-align: left;
}}
#loadBtn:hover {{
    background-color: {ACCENT_HOVER};
}}

#newChatBtn {{
    background-color: transparent;
    color: {TEXT_MUTED};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 9px 16px;
    font-size: 13px;
    margin: 0 16px 8px 16px;
    text-align: left;
}}
#newChatBtn:hover {{
    background-color: {BUBBLE_AI};
    color: {TEXT_PRIMARY};
}}

#statusDot {{
    font-size: 12px;
    padding: 20px 20px 20px 20px;
}}

/* ── Основная область чата ── */
#chatArea {{
    background-color: {CHAT_BG};
}}

#scrollArea {{
    background-color: {CHAT_BG};
    border: none;
}}

QScrollBar:vertical {{
    background: transparent;
    width: 4px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 2px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {TEXT_MUTED};
}}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0;
    background: transparent;
}}
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {{
    background: transparent;
}}


#scrollContent {{
    background-color: {CHAT_BG};
}}

/* ── Пузыри сообщений ── */
#bubbleUser {{
    background-color: {BUBBLE_USER};
    border-radius: 12px;
    padding: 12px 16px;
    color: {TEXT_PRIMARY};
    font-size: 14px;
    line-height: 1.6;
    max-width: 640px;
}}

#bubbleAI {{
    background-color: transparent;
    border-radius: 12px;
    padding: 12px 16px;
    color: {TEXT_PRIMARY};
    font-size: 14px;
    line-height: 1.6;
    max-width: 680px;
}}

#roleLabel {{
    font-size: 11px;
    font-weight: 600;
    color: {TEXT_MUTED};
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}}

/* ── Поле ввода ── */
#inputContainer {{
    background-color: {SIDEBAR_BG};
    border-top: 1px solid {BORDER};
    padding: 5px 20px;
}}

#inputField {{
    background-color: {INPUT_BG};
    border: 1px solid {BORDER};
    border-radius: 12px;
    color: {TEXT_PRIMARY};
    font-size: 14px;
    padding: 5px 16px;
    min-height: 24px;
    max-height: 80px;
}}
#inputField:focus {{
    border: 1px solid {ACCENT};
    outline: none;
}}

#sendBtn {{
    background-color: {ACCENT};
    color: white;
    border: none;
    border-radius: 10px;
    min-width: 48px;
    min-height: 48px;
    font-size: 18px;
    font-weight: bold;
    margin: 0 10px;
}}
#sendBtn:hover {{
    background-color: {ACCENT_HOVER};
}}
#sendBtn:disabled {{
    background-color: {BORDER};
    color: {TEXT_MUTED};
}}

#emptyState {{
    color: {TEXT_MUTED};
    font-size: 15px;
}}
"""



class InferenceWorker(QObject):
    token_ready   = Signal(str)
    finished      = Signal()
    error_occurred = Signal(str)

    def __init__(self, llm, messages):
        super().__init__()
        self.llm      = llm
        self.messages = messages

    def run(self):
        try:
            stream = self.llm.create_chat_completion(
                messages=self.messages,
                stream=True,
                max_tokens=1024,
                temperature=0.7,
            )
            for chunk in stream:
                delta = chunk["choices"][0]["delta"]
                if "content" in delta and delta["content"]:
                    self.token_ready.emit(delta["content"])
            self.finished.emit()
        except Exception as e:
            self.error_occurred.emit(str(e))



class MessageBubble(QWidget):
    def __init__(self, role: str, parent=None):
        super().__init__(parent)
        self.role = role

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 8, 24, 8)
        outer.setSpacing(4)

        role_lbl = QLabel("Вы" if role == "user" else "Ассистент")
        role_lbl.setObjectName("roleLabel")
        outer.addWidget(role_lbl)

        # Текст
        self.text_label = QLabel()
        self.text_label.setObjectName("bubbleUser" if role == "user" else "bubbleAI")
        self.text_label.setWordWrap(True)
        self.text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.text_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        if role == "user":
            row = QHBoxLayout()
            row.addStretch()
            row.addWidget(self.text_label)
            outer.addLayout(row)
        else:
            outer.addWidget(self.text_label)

    def set_text(self, text: str):
        self.text_label.setText(text)

    def append_text(self, token: str):
        self.text_label.setText(self.text_label.text() + token)



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LocalAI")
        self.resize(1100, 720)
        self.setMinimumSize(800, 560)

        self.llm           = None
        self.model_path    = None
        self.history       = []
        self.current_bubble = None
        self.worker        = None
        self.thread        = None

        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._make_sidebar())
        layout.addWidget(self._make_chat_panel(), stretch=1)

    def _make_sidebar(self):
        sb = QWidget()
        sb.setObjectName("sidebar")
        vl = QVBoxLayout(sb)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        title = QLabel("LocalAI")
        title.setObjectName("appTitle")
        vl.addWidget(title)

        load_btn = QPushButton("＋  Загрузить модель")
        load_btn.setObjectName("loadBtn")
        load_btn.clicked.connect(self._on_load_model)
        vl.addWidget(load_btn)

        new_btn = QPushButton("✦  Новый чат")
        new_btn.setObjectName("newChatBtn")
        new_btn.clicked.connect(self._on_new_chat)
        vl.addWidget(new_btn)

        lbl = QLabel("МОДЕЛЬ")
        lbl.setObjectName("modelLabel")
        vl.addWidget(lbl)

        self.model_name_lbl = QLabel("не загружена")
        self.model_name_lbl.setObjectName("modelName")
        self.model_name_lbl.setWordWrap(True)
        vl.addWidget(self.model_name_lbl)

        vl.addStretch()

        self.status_lbl = QLabel("⬤  Не активно")
        self.status_lbl.setObjectName("statusDot")
        self.status_lbl.setStyleSheet(f"color: {TEXT_MUTED};")
        vl.addWidget(self.status_lbl)

        return sb

    def _make_chat_panel(self):
        panel = QWidget()
        panel.setObjectName("chatArea")
        vl = QVBoxLayout(panel)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        # Скролл с сообщениями
        self.scroll = QScrollArea()
        self.scroll.setObjectName("scrollArea")
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("scrollContent")
        self.messages_layout = QVBoxLayout(self.scroll_content)
        self.messages_layout.setContentsMargins(0, 16, 0, 16)
        self.messages_layout.setSpacing(4)

        self.empty_label = QLabel("Загрузите модель и начните диалог")
        self.empty_label.setObjectName("emptyState")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.messages_layout.addWidget(self.empty_label)
        self.messages_layout.addStretch()

        self.scroll.setWidget(self.scroll_content)
        vl.addWidget(self.scroll, stretch=1)

        vl.addWidget(self._make_input_bar())
        return panel

    def _make_input_bar(self):
        container = QWidget()
        container.setObjectName("inputContainer")
        hl = QHBoxLayout(container)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(0)

        self.input_field = QTextEdit()
        self.input_field.setObjectName("inputField")
        self.input_field.setPlaceholderText("Написать сообщение…")
        self.input_field.setFixedHeight(52)
        self.input_field.installEventFilter(self)

        self.send_btn = QPushButton("↑")
        self.send_btn.setObjectName("sendBtn")
        self.send_btn.setFixedSize(52, 52)
        self.send_btn.clicked.connect(self._on_send)
        self.send_btn.setEnabled(False)

        hl.addWidget(self.input_field)
        hl.addWidget(self.send_btn)
        return container

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QKeyEvent
        if obj is self.input_field and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Return and not (event.modifiers() & Qt.ShiftModifier):
                self._on_send()
                return True
        return super().eventFilter(obj, event)

    def _on_load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите GGUF модель", "", "GGUF Models (*.gguf);;All Files (*)"
        )
        if not path:
            return

        self.model_path = path
        name = Path(path).name
        self.model_name_lbl.setText(name)
        self.status_lbl.setText("⬤  Загружается…")
        self.status_lbl.setStyleSheet(f"color: #f0a500; padding: 20px 20px 20px 20px;")
        self.send_btn.setEnabled(False)

        thread = threading.Thread(target=self._load_model_thread, args=(path,), daemon=True)
        thread.start()

    def _load_model_thread(self, path: str):
        try:
            if LLAMA_AVAILABLE:
                try:
                    self.llm = Llama(
                        model_path=path,
                        n_ctx=4096,
                        n_threads=4,
                        n_gpu_layers=0,
                        verbose=True,
                    )
                except Exception as e:
                    self.llm = Llama(
                        model_path=path,
                        n_ctx=4096,
                        n_threads=4,
                        n_gpu_layers=0,
                        verbose=True,
                    )
            from PySide6.QtCore import QMetaObject, Q_ARG
            QMetaObject.invokeMethod(self, "_on_model_loaded", Qt.QueuedConnection)
        except Exception as e:
            from PySide6.QtCore import QMetaObject
            QMetaObject.invokeMethod(
                self, "_on_model_error",
                Qt.QueuedConnection,
                Q_ARG(str, str(e))
            )

    @Slot()
    def _on_model_loaded(self):
        self.status_lbl.setText("⬤  Готово")
        self.status_lbl.setStyleSheet(f"color: #4ade80; padding: 0 20px 20px 20px;")
        self.send_btn.setEnabled(True)
        self.empty_label.setText("Модель загружена. Начните диалог ↓")
    @Slot()
    def _on_model_error(self, msg: str):
        self.status_lbl.setText("⬤  Ошибка загрузки")
        self.status_lbl.setStyleSheet(f"color: #f87171; padding: 0 20px 20px 20px;")
        self.empty_label.setText(f"Ошибка: {msg}")

    @Slot()
    def _on_new_chat(self):
        self.history.clear()
        while self.messages_layout.count() > 0:
            item = self.messages_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.empty_label = QLabel(
            "Новый чат начат" if self.llm else "Загрузите модель и начните диалог"
        )
        self.empty_label.setObjectName("emptyState")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.messages_layout.addWidget(self.empty_label)
        self.messages_layout.addStretch()

    def _on_send(self):
        text = self.input_field.toPlainText().strip()
        if not text:
            return
        if not LLAMA_AVAILABLE or not self.llm:
            self._add_bubble("user", text)
            self._add_bubble("assistant", "[Модель не загружена — демо-режим]")
            self.input_field.clear()
            return

        self.input_field.clear()
        self.send_btn.setEnabled(False)

        if self.empty_label:
            self.empty_label.deleteLater()
            self.empty_label = None


            for i in range(self.messages_layout.count()):
                item = self.messages_layout.itemAt(i)
                if item and item.spacerItem():
                    self.messages_layout.removeItem(item)
                    break

        self.history.append({"role": "user", "content": text})
        self._add_bubble("user", text)

        self.current_bubble = self._add_bubble("assistant", "")

        self.thread = QThread()
        self.worker = InferenceWorker(self.llm, list(self.history))
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.token_ready.connect(self._on_token)
        self.worker.finished.connect(self._on_inference_done)
        self.worker.error_occurred.connect(self._on_inference_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _add_bubble(self, role: str, text: str) -> MessageBubble:
        bubble = MessageBubble(role)
        bubble.set_text(text)
        count = self.messages_layout.count()
        self.messages_layout.insertWidget(count, bubble)
        self._scroll_to_bottom()
        return bubble

    def _on_token(self, token: str):
        if self.current_bubble:
            self.current_bubble.append_text(token)
            self._scroll_to_bottom()

    def _on_inference_done(self):
        if self.current_bubble:
            full_text = self.current_bubble.text_label.text()
            self.history.append({"role": "assistant", "content": full_text})
        self.current_bubble = None
        self.send_btn.setEnabled(True)

    def _on_inference_error(self, msg: str):
        if self.current_bubble:
            self.current_bubble.set_text(f"[Ошибка: {msg}]")
        self.current_bubble = None
        self.send_btn.setEnabled(True)

    def _scroll_to_bottom(self):
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())



def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
