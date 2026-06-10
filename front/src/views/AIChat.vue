<template>
  <div class="chat-cockpit" :class="{ 'chat-cockpit--evidence-open': showEvidencePanel }">
    <aside class="conversation-rail enterprise-card">
      <button class="enterprise-button new-chat-button" @click="startBlankSession">+ 新建会话</button>
      <input v-model="sessionSearch" class="enterprise-input" placeholder="搜索历史会话" />

      <div class="rail-section-title">最近会话</div>
      <div class="session-list-scroll">
        <div v-if="sessionStore.isLoading" class="rail-empty rail-loading" role="status" aria-live="polite">
          <span class="rail-loading-dot"></span>
          正在加载会话...
        </div>
        <template v-else>
          <button
            v-for="session in filteredSessions"
            :key="session.session_id"
            class="session-row"
            :class="{ active: session.session_id === sessionId }"
            @click="selectSession(session)"
          >
            <strong>{{ session.title || '新会话' }}</strong>
            <span>{{ formatSessionTime(session.updated_at || session.created_at) }}</span>
          </button>

          <div v-if="filteredSessions.length === 0" class="rail-empty">
            暂无匹配会话
          </div>
        </template>
      </div>
    </aside>

    <section class="chat-workspace enterprise-card">
      <header class="chat-header">
        <div>
          <p class="shell-eyebrow">Knowledge Assistant</p>
          <h2>{{ currentSessionTitle }}</h2>
        </div>
        <div class="chat-header-tags">
          <span class="enterprise-tag">SSE 流式输出</span>
          <span class="enterprise-tag">Markdown</span>
          <span class="enterprise-tag">代码高亮</span>
          <button class="evidence-toggle" type="button" :aria-expanded="showEvidencePanel" @click="showEvidencePanel = !showEvidencePanel">
            {{ showEvidencePanel ? '关闭引用' : '引用证据' }}
          </button>
        </div>
      </header>

      <div class="messages-container" ref="messagesContainer">
        <div v-if="isWelcomeState" class="welcome-panel">
          <span class="enterprise-tag">企业知识问答</span>
          <h2>从内部知识库中获取可追溯答案</h2>
          <p>选择一个推荐问题，或直接输入你的业务问题。</p>
          <div class="prompt-grid">
            <button v-for="prompt in recommendedPrompts" :key="prompt" @click="usePrompt(prompt)">
              {{ prompt }}
            </button>
          </div>
        </div>

        <div
          v-for="(message, index) in messages"
          :key="index"
          :class="['message', message.role === 'user' ? 'user-message' : 'ai-message']"
        >
          <div class="message-avatar">{{ message.role === 'user' ? '我' : 'AI' }}</div>
          <div class="message-content">
            <div v-if="message.role === 'assistant' && message.content === ''" class="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
            <div v-else v-html="formatMessage(message.content)"></div>
          </div>
        </div>
      </div>

      <footer class="chat-composer">
        <textarea
          v-model="userInput"
          class="enterprise-textarea"
          placeholder="请输入企业知识库问题，Enter 发送，Shift + Enter 换行"
          @keydown.enter.exact.prevent="sendMessage"
        ></textarea>
        <button class="enterprise-button" :disabled="isLoading || !userInput.trim()" @click="sendMessage">
          {{ isLoading ? '生成中...' : '发送' }}
        </button>
      </footer>
    </section>

    <aside v-if="showEvidencePanel" class="evidence-panel enterprise-card">
      <header>
        <p class="shell-eyebrow">Retrieval Evidence</p>
        <h3>引用与检索证据</h3>
      </header>

      <div v-if="evidenceItems.length === 0" class="evidence-empty">
        <strong>本轮暂未返回引用信息</strong>
        <span>后端返回引用字段后，可在这里展示来源文档、召回片段和重排序结果。</span>
      </div>

      <article v-for="item in evidenceItems" :key="item.title" class="evidence-card">
        <div>
          <strong>{{ item.title }}</strong>
          <span>{{ item.source }}</span>
        </div>
        <p>{{ item.summary }}</p>
        <div class="evidence-tags">
          <span class="enterprise-tag">{{ item.method }}</span>
          <span class="enterprise-tag">Score {{ item.score }}</span>
        </div>
      </article>
    </aside>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, nextTick, watch } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { showToast } from 'vant';
import { marked } from 'marked';
import { markedHighlight } from 'marked-highlight';
import DOMPurify from 'dompurify';
import hljs from 'highlight.js';
import 'highlight.js/styles/monokai-sublime.css';
import 'highlight.js/lib/common';
import { useUserStore } from '../store/user';
import { useSessionStore } from '../store/session';

// 从cookie中获取CSRF token
const getCsrfToken = () => {
  const cookieValue = document.cookie
    .split('; ')
    .find(row => row.startsWith('csrftoken='))
    ?.split('=')[1];
  return cookieValue || '';
};

// 聊天消息
const messages = ref([
  { role: 'assistant', content: '你好！我是AI助手，有什么可以帮助你的吗？' }
]);
const userInput = ref('');
const messagesContainer = ref(null);
const isLoading = ref(false);
const sessionId = ref('');
const hasJumped = ref(false);
const sessionSearch = ref('');
const recommendedPrompts = [
  '公司报销流程是什么？',
  '如何申请年假？',
  '产品资料在哪里？',
  '知识库中有哪些安全规范？'
];
const evidenceItems = ref([]);
const showEvidencePanel = ref(false);

const router = useRouter();
const route = useRoute();
const userStore = useUserStore();
const sessionStore = useSessionStore();

const filteredSessions = computed(() => {
  const keyword = sessionSearch.value.trim().toLowerCase();
  if (!keyword) return sessionStore.sessions;
  return sessionStore.sessions.filter((session) => (session.title || '新会话').toLowerCase().includes(keyword));
});

const currentSessionTitle = computed(() => {
  const current = sessionStore.sessions.find((session) => session.session_id === sessionId.value);
  return current?.title || '新的知识库问答';
});

const isWelcomeState = computed(() => messages.value.length === 1 && messages.value[0]?.role === 'assistant');

const formatSessionTime = (timeString) => {
  if (!timeString) return '刚刚';
  return new Date(timeString).toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  });
};

const startBlankSession = () => {
  sessionId.value = '';
  evidenceItems.value = [];
  messages.value = [{ role: 'assistant', content: '你好！我是AI助手，有什么可以帮助你的吗？' }];
  router.push('/aichat');
};

const selectSession = (session) => {
  router.push(`/aichat/${session.session_id}`);
};

const usePrompt = (prompt) => {
  userInput.value = prompt;
  sendMessage();
};

// 配置marked使用marked-highlight插件
marked.use(markedHighlight({
  langPrefix: 'hljs language-',
  highlight(code, lang) {
    const language = hljs.getLanguage(lang) ? lang : 'plaintext';
    return hljs.highlight(code, { language }).value;
  }
}));

// 格式化消息内容（支持Markdown和代码高亮）
const formatMessage = (content) => {
  if (!content) return '';
  try {
    // 使用marked解析Markdown，并用DOMPurify清理HTML
    const parsed = marked(content, {
      breaks: true,
      gfm: true,
      headerIds: false,
      mangle: false
    });
    const sanitized = DOMPurify.sanitize(parsed);
    return sanitized;
  } catch (error) {
    console.error('Markdown解析错误:', error);
    return content;
  }
};

// 发送消息
const sendMessage = async () => {
  if (!userInput.value.trim() || isLoading.value) return;

  // 检查是否登录
  if (!userStore.getLoginStatus) {
    showToast('请先登录');
    return;
  }

  // 添加用户消息
  const userMessage = userInput.value.trim();
  messages.value.push({ role: 'user', content: userMessage });
  userInput.value = '';

  // 添加AI消息占位
  messages.value.push({ role: 'assistant', content: '' });
  evidenceItems.value = [];

  // 滚动到底部
  await nextTick();
  scrollToBottom();

  // 发送请求
  isLoading.value = true;
  try {
    await fetchAIResponse(userMessage);
  } catch (error) {
    console.error('Error fetching AI response:', error);
    // 更新最后一条消息为错误信息
    messages.value[messages.value.length - 1].content = `发生错误: ${error.message || '请检查网络连接和API设置'}`;
  } finally {
    isLoading.value = false;
    await nextTick();
    scrollToBottom();
  }
};

// 获取AI响应（使用SSE）
const fetchAIResponse = async (userMessage) => {
  try {
    // 确保使用正确的相对路径，通过Vite代理访问
    const url = '/api/agent/query/stream';
    // 从localStorage获取token
    const token = localStorage.getItem('jwt_token') || userStore.token;
    // console.log('发送AI请求到:', url);
    // console.log('使用的token:', token);

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({
        session_id: sessionId.value || undefined,
        query: userMessage
      })
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }

    // 处理SSE流
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let aiResponse = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6);
        if (!data) continue;

        try {
          const json = JSON.parse(data);

          switch (json.type) {
            case 'step':
              break;
            case 'response':
              const content = json.content || '';
              if (content) {
                aiResponse += content;

                // 逐字符显示打字机效果
                const displayContent = messages.value[messages.value.length - 1].content || '';
                const remainingContent = aiResponse.substring(displayContent.length);

                for (const char of remainingContent) {
                  messages.value[messages.value.length - 1].content += char;
                  await nextTick();
                  scrollToBottom();
                  // 控制打字速度，每个字符延迟8ms
                  await new Promise(resolve => setTimeout(resolve, 8));
                }
              }
              // 保存会话ID（不立即跳转，避免中断SSE）
              if (json.session_id && typeof json.session_id === 'string' && json.session_id.trim()) {
                sessionId.value = json.session_id;
              }
              break;
            case 'done':
              // 保存会话ID并在所有数据接收完成后跳转
              if (json.session_id && typeof json.session_id === 'string' && json.session_id.trim()) {
                sessionId.value = json.session_id;
                // 如果当前路由没有sessionId参数，跳转到带sessionId的路由
                if (!route.params.sessionId) {
                  router.push(`/aichat/${json.session_id}`);
                }
              }
              break;
            case 'error':
              throw new Error(json.content || 'API错误');
              break;
          }
        } catch (e) {
          console.error('Error parsing SSE data:', e);
        }
      }
    }
  }

  // 如果没有收到任何内容
  if (!aiResponse) {
    messages.value[messages.value.length - 1].content = '抱歉，我无法生成回复。请检查API设置或稍后再试。';
  }
  } catch (error) {
    console.error('Fetch error:', error);
    throw error;
  }
};

// 滚动到底部
const scrollToBottom = () => {
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight;
  }
};

// 监听消息变化，自动滚动
watch(messages, () => {
  nextTick(() => {
    scrollToBottom();
  });
}, { deep: true });

// 监听路由参数变化，重新加载会话历史
watch(() => route.params.sessionId, async (newSessionId) => {
  if (newSessionId) {
    try {
      const result = await sessionStore.getSession(newSessionId);
      if (result.success && sessionStore.currentSession) {
        loadSessionHistory(sessionStore.currentSession);
      } else {
        showToast('加载会话历史失败');
      }
    } catch (error) {
      console.error('加载会话历史失败:', error);
      showToast('加载会话历史失败');
    }
  }
}, { immediate: true });

const loadSessionRail = async () => {
  if (!userStore.getLoginStatus) return;
  if (!userStore.userInfo) {
    await userStore.getUserInfoDetail();
  }
  const userId = userStore.userInfo?.uuid || userStore.userInfo?.id || userStore.userInfo?.user_id;
  if (userId) {
    await sessionStore.getUserSessions(userId);
  }
};

// 组件挂载时检查是否有当前会话或路由参数中的会话ID
onMounted(async () => {
  await loadSessionRail();
  // 检查路由参数中是否有sessionId
  const routeSessionId = route.params.sessionId;

  if (routeSessionId) {
    // 从路由参数获取会话ID，加载会话历史
    try {
      const result = await sessionStore.getSession(routeSessionId);
      if (result.success && sessionStore.currentSession) {
        loadSessionHistory(sessionStore.currentSession);
      } else {
        showToast('加载会话历史失败');
      }
    } catch (error) {
      console.error('加载会话历史失败:', error);
      showToast('加载会话历史失败');
    }
  } else if (sessionStore.currentSession) {
    // 从store中加载会话历史
    loadSessionHistory(sessionStore.currentSession);
  }

  scrollToBottom();
});

// 加载会话历史
const loadSessionHistory = (session) => {
  if (session.history && session.history.length > 0) {
    // 清空当前消息
    messages.value = [];
    // 加载历史消息
    session.history.forEach(([userMsg, aiMsg]) => {
      messages.value.push({ role: 'user', content: userMsg });
      messages.value.push({ role: 'assistant', content: aiMsg });
    });
    // 设置会话ID
    sessionId.value = session.session_id;
  }
};
</script>

<style scoped>
.chat-cockpit {
  display: grid;
  grid-template-columns: 280px minmax(420px, 1fr);
  gap: 20px;
  height: calc(100vh - 142px);
  min-height: 680px;
  color: #172033;
}

.chat-cockpit--evidence-open {
  grid-template-columns: 280px minmax(420px, 1fr) 320px;
}

.enterprise-card {
  background: rgba(255, 255, 255, 0.94);
  border: 1px solid rgba(148, 163, 184, 0.22);
  border-radius: 24px;
  box-shadow: 0 22px 60px rgba(15, 23, 42, 0.10);
  backdrop-filter: blur(18px);
}

.enterprise-button {
  border: 0;
  border-radius: 14px;
  background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 55%, #0f766e 100%);
  color: #fff;
  font-weight: 800;
  letter-spacing: 0.02em;
  cursor: pointer;
  box-shadow: 0 14px 28px rgba(37, 99, 235, 0.24);
  transition: transform 0.18s ease, box-shadow 0.18s ease, opacity 0.18s ease;
}

.enterprise-button:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 18px 36px rgba(37, 99, 235, 0.32);
}

.enterprise-button:disabled {
  cursor: not-allowed;
  opacity: 0.48;
  box-shadow: none;
}

.enterprise-input,
.enterprise-textarea {
  width: 100%;
  border: 1px solid rgba(148, 163, 184, 0.34);
  border-radius: 14px;
  background: #f8fafc;
  color: #0f172a;
  outline: none;
  transition: border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
}

.enterprise-input:focus,
.enterprise-textarea:focus {
  border-color: rgba(37, 99, 235, 0.72);
  background: #fff;
  box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.10);
}

.enterprise-tag {
  display: inline-flex;
  align-items: center;
  border: 1px solid rgba(37, 99, 235, 0.16);
  border-radius: 999px;
  background: rgba(37, 99, 235, 0.08);
  color: #1d4ed8;
  font-size: 12px;
  font-weight: 800;
  line-height: 1;
  padding: 7px 10px;
}

.shell-eyebrow {
  margin: 0 0 8px;
  color: #2563eb;
  font-size: 11px;
  font-weight: 900;
  letter-spacing: 0.18em;
  text-transform: uppercase;
}

.conversation-rail {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-width: 0;
  padding: 18px;
  overflow: hidden;
}

.new-chat-button {
  width: 100%;
  padding: 13px 16px;
  text-align: center;
}

.conversation-rail .enterprise-input {
  flex: 0 0 auto;
  padding: 12px 14px;
  font-size: 14px;
}

.rail-section-title {
  margin-top: 8px;
  color: #64748b;
  font-size: 12px;
  font-weight: 900;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.session-list-scroll {
  display: grid;
  flex: 1 1 auto;
  gap: 12px;
  min-height: 0;
  overflow-y: auto;
}

.session-row {
  display: grid;
  gap: 6px;
  width: 100%;
  border: 1px solid transparent;
  border-radius: 16px;
  background: transparent;
  padding: 13px 14px;
  color: #334155;
  text-align: left;
  cursor: pointer;
  transition: background 0.18s ease, border-color 0.18s ease, transform 0.18s ease;
}

.session-row:hover,
.session-row.active {
  border-color: rgba(37, 99, 235, 0.18);
  background: linear-gradient(135deg, rgba(37, 99, 235, 0.10), rgba(14, 165, 233, 0.06));
  transform: translateX(2px);
}

.session-row strong {
  overflow: hidden;
  color: #0f172a;
  font-size: 14px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.session-row span,
.rail-empty {
  color: #64748b;
  font-size: 12px;
}

.rail-empty {
  border: 1px dashed rgba(148, 163, 184, 0.42);
  border-radius: 16px;
  padding: 18px;
  text-align: center;
}

.rail-loading {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
}

.rail-loading-dot {
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: #2563eb;
  box-shadow: 0 0 0 6px rgba(37, 99, 235, 0.12);
}

.chat-workspace {
  display: grid;
  grid-template-rows: auto minmax(0, 1fr) auto;
  min-width: 0;
  overflow: hidden;
}

.chat-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 18px;
  border-bottom: 1px solid rgba(148, 163, 184, 0.22);
  padding: 22px 24px;
  background:
    radial-gradient(circle at top left, rgba(37, 99, 235, 0.10), transparent 34%),
    linear-gradient(180deg, rgba(248, 250, 252, 0.96), rgba(255, 255, 255, 0.72));
}

.chat-header h2,
.evidence-panel h3,
.welcome-panel h2 {
  margin: 0;
  color: #0f172a;
  letter-spacing: -0.03em;
}

.chat-header h2 {
  font-size: 24px;
  line-height: 1.16;
}

.chat-header-tags {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
}

.evidence-toggle {
  border: 1px solid rgba(37, 99, 235, 0.22);
  border-radius: 999px;
  background: #fff;
  color: #1d4ed8;
  cursor: pointer;
  font-size: 12px;
  font-weight: 900;
  padding: 7px 12px;
  transition: border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease;
}

.evidence-toggle:hover {
  border-color: rgba(37, 99, 235, 0.45);
  box-shadow: 0 10px 22px rgba(37, 99, 235, 0.14);
  transform: translateY(-1px);
}

.messages-container {
  min-height: 0;
  overflow-y: auto;
  padding: 24px;
  background:
    linear-gradient(rgba(15, 23, 42, 0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(15, 23, 42, 0.035) 1px, transparent 1px),
    #f8fafc;
  background-size: 28px 28px;
  scroll-behavior: smooth;
}

.messages-container::-webkit-scrollbar,
.session-list-scroll::-webkit-scrollbar,
.evidence-panel::-webkit-scrollbar {
  width: 10px;
}

.messages-container::-webkit-scrollbar-thumb,
.session-list-scroll::-webkit-scrollbar-thumb,
.evidence-panel::-webkit-scrollbar-thumb {
  border: 3px solid transparent;
  border-radius: 999px;
  background: rgba(100, 116, 139, 0.34);
  background-clip: content-box;
}

.welcome-panel {
  position: relative;
  overflow: hidden;
  border: 1px solid rgba(37, 99, 235, 0.14);
  border-radius: 24px;
  margin-bottom: 24px;
  padding: 28px;
  background:
    linear-gradient(135deg, rgba(239, 246, 255, 0.95), rgba(255, 255, 255, 0.84)),
    radial-gradient(circle at 92% 20%, rgba(14, 165, 233, 0.20), transparent 28%);
  box-shadow: 0 18px 48px rgba(37, 99, 235, 0.12);
}

.welcome-panel::after {
  position: absolute;
  right: -48px;
  bottom: -72px;
  width: 190px;
  height: 190px;
  border: 28px solid rgba(37, 99, 235, 0.08);
  border-radius: 50%;
  content: '';
}

.welcome-panel h2 {
  max-width: 560px;
  margin-top: 14px;
  font-size: 30px;
}

.welcome-panel p {
  max-width: 520px;
  margin: 12px 0 20px;
  color: #475569;
  line-height: 1.7;
}

.prompt-grid {
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.prompt-grid button {
  border: 1px solid rgba(37, 99, 235, 0.14);
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.84);
  color: #1e293b;
  cursor: pointer;
  font-weight: 800;
  padding: 14px 16px;
  text-align: left;
  transition: border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease;
}

.prompt-grid button:hover {
  border-color: rgba(37, 99, 235, 0.38);
  box-shadow: 0 14px 30px rgba(37, 99, 235, 0.14);
  transform: translateY(-2px);
}

.message {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  width: min(820px, 100%);
  margin-bottom: 18px;
}

.user-message {
  flex-direction: row-reverse;
  margin-left: auto;
}

.message-avatar {
  display: grid;
  flex: 0 0 40px;
  width: 40px;
  height: 40px;
  place-items: center;
  border-radius: 14px;
  background: #0f172a;
  color: #fff;
  font-size: 13px;
  font-weight: 900;
  box-shadow: 0 12px 26px rgba(15, 23, 42, 0.20);
}

.user-message .message-avatar {
  background: linear-gradient(135deg, #1d4ed8, #0f766e);
}

.message-content {
  min-width: 0;
  max-width: 100%;
  border: 1px solid rgba(148, 163, 184, 0.20);
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.94);
  box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
  color: #1e293b;
  line-height: 1.72;
  padding: 14px 16px;
  word-break: break-word;
}

.user-message .message-content {
  border-color: rgba(37, 99, 235, 0.18);
  background: linear-gradient(135deg, #1d4ed8, #2563eb);
  color: #fff;
}

.chat-composer {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 118px;
  gap: 14px;
  border-top: 1px solid rgba(148, 163, 184, 0.22);
  padding: 18px 20px;
  background: rgba(255, 255, 255, 0.94);
}

.enterprise-textarea {
  min-height: 76px;
  max-height: 180px;
  padding: 14px 16px;
  font: inherit;
  resize: vertical;
}

.chat-composer .enterprise-button {
  min-height: 76px;
  padding: 0 18px;
}

.typing-indicator {
  display: flex;
  align-items: center;
  min-height: 26px;
  padding: 3px;
}

.typing-indicator span {
  display: inline-block;
  width: 8px;
  height: 8px;
  margin: 0 3px;
  border-radius: 50%;
  background: #2563eb;
  animation: bounce 1.5s infinite ease-in-out;
}

.typing-indicator span:nth-child(2) {
  animation-delay: 0.2s;
}

.typing-indicator span:nth-child(3) {
  animation-delay: 0.4s;
}

.evidence-panel {
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding: 20px;
  overflow-y: auto;
}

.evidence-panel header {
  border-bottom: 1px solid rgba(148, 163, 184, 0.22);
  padding-bottom: 14px;
}

.evidence-panel h3 {
  font-size: 20px;
}

.evidence-empty {
  display: grid;
  gap: 8px;
  border: 1px dashed rgba(37, 99, 235, 0.30);
  border-radius: 18px;
  background: rgba(239, 246, 255, 0.70);
  color: #475569;
  padding: 18px;
}

.evidence-empty strong {
  color: #0f172a;
}

.evidence-card {
  display: grid;
  gap: 12px;
  border: 1px solid rgba(148, 163, 184, 0.22);
  border-radius: 18px;
  background: linear-gradient(180deg, #fff, #f8fafc);
  padding: 16px;
  box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
}

.evidence-card div:first-child {
  display: grid;
  gap: 4px;
}

.evidence-card strong {
  color: #0f172a;
}

.evidence-card span,
.evidence-card p {
  color: #64748b;
}

.evidence-card p {
  margin: 0;
  line-height: 1.6;
}

.evidence-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

@keyframes bounce {
  0%, 60%, 100% {
    transform: translateY(0);
  }
  30% {
    transform: translateY(-6px);
  }
}

/* Markdown 样式 */
:deep(pre) {
  overflow-x: auto;
  border: 1px solid rgba(148, 163, 184, 0.24);
  border-radius: 14px;
  background-color: #101827;
  color: #d4d4d4;
  margin: 12px 0;
  padding: 16px;
}

:deep(pre code) {
  background-color: transparent;
  border-radius: 0;
  color: #d4d4d4;
  padding: 0;
}

:deep(code) {
  border-radius: 6px;
  background-color: rgba(15, 23, 42, 0.08);
  color: inherit;
  font-family: 'JetBrains Mono', 'Cascadia Code', 'Consolas', monospace;
  font-size: 0.9em;
  padding: 2px 6px;
}

.user-message :deep(code) {
  background-color: rgba(255, 255, 255, 0.16);
}

:deep(p) {
  margin: 8px 0;
  line-height: 1.7;
}

:deep(ul),
:deep(ol) {
  margin: 8px 0;
  padding-left: 22px;
}

:deep(li) {
  margin: 4px 0;
  line-height: 1.6;
}

:deep(a) {
  color: #2563eb;
  font-weight: 700;
  text-decoration: none;
}

.user-message :deep(a) {
  color: #bfdbfe;
}

:deep(a:hover) {
  text-decoration: underline;
}

:deep(h1),
:deep(h2),
:deep(h3),
:deep(h4),
:deep(h5),
:deep(h6) {
  margin: 14px 0 8px;
  color: inherit;
  font-weight: 900;
  line-height: 1.24;
}

:deep(h1) {
  font-size: 1.5em;
}

:deep(h2) {
  font-size: 1.3em;
}

:deep(h3) {
  font-size: 1.12em;
}

:deep(blockquote) {
  border-left: 4px solid #2563eb;
  border-radius: 0 12px 12px 0;
  background-color: rgba(37, 99, 235, 0.08);
  color: #475569;
  margin: 12px 0;
  padding: 10px 14px;
}

.user-message :deep(blockquote) {
  background-color: rgba(255, 255, 255, 0.14);
  color: #eff6ff;
}

:deep(hr) {
  border: 0;
  border-top: 1px solid rgba(148, 163, 184, 0.30);
  margin: 16px 0;
}

:deep(img) {
  max-width: 100%;
  border-radius: 12px;
  margin: 8px 0;
}

:deep(table) {
  width: 100%;
  border-collapse: collapse;
  margin: 12px 0;
  overflow: hidden;
  border-radius: 12px;
}

:deep(th),
:deep(td) {
  border: 1px solid rgba(148, 163, 184, 0.32);
  padding: 9px;
  text-align: left;
}

:deep(th) {
  background-color: rgba(37, 99, 235, 0.08);
  font-weight: 900;
}

@media (max-width: 1200px) {
  .chat-cockpit {
    grid-template-columns: 260px minmax(0, 1fr);
  }
}

@media (max-width: 820px) {
  .chat-cockpit {
    grid-template-columns: 1fr;
    height: auto;
    min-height: 0;
  }

  .conversation-rail,
  .chat-workspace {
    min-height: auto;
  }

  .chat-workspace {
    height: min(760px, calc(100vh - 120px));
  }

  .chat-header,
  .chat-composer {
    grid-template-columns: 1fr;
  }

  .chat-header {
    flex-direction: column;
  }

  .chat-header-tags {
    justify-content: flex-start;
  }

  .prompt-grid {
    grid-template-columns: 1fr;
  }

  .message {
    width: 100%;
  }

  .chat-composer {
    display: grid;
  }

  .chat-composer .enterprise-button {
    min-height: 48px;
  }
}
</style>
