<template>
  <div class="sessions-page">
    <section class="sessions-hero enterprise-card">
      <div>
        <p class="shell-eyebrow">Conversation Management</p>
        <h2>会话管理</h2>
        <p>搜索、筛选并继续你的企业知识库问答。</p>
      </div>
      <button class="enterprise-button" @click="createNewSession">新建会话</button>
    </section>

    <section class="sessions-toolbar enterprise-card">
      <input v-model="searchKeyword" class="enterprise-input" placeholder="搜索会话标题" />
      <div class="session-filters">
        <button v-for="filter in filters" :key="filter.value" :class="{ active: activeFilter === filter.value }" @click="activeFilter = filter.value">
          {{ filter.label }}
        </button>
      </div>
    </section>

    <section v-if="sessionStore.isLoading" class="sessions-state enterprise-card">
      <van-loading type="spinner" color="#2563eb" />
      <p>加载中...</p>
    </section>

    <section v-else-if="filteredSessions.length === 0" class="sessions-state enterprise-card">
      <h3>暂无会话记录</h3>
      <p>创建一个新会话，开始企业知识库问答。</p>
      <button class="enterprise-button" @click="createNewSession">创建新会话</button>
    </section>

    <section v-else class="sessions-grid">
      <article v-for="session in filteredSessions" :key="session.session_id" class="session-card enterprise-card">
        <div class="session-card-header">
          <span class="enterprise-tag">{{ sessionStatus(session) }}</span>
          <button class="enterprise-button danger" @click.stop="deleteSession(session.session_id)">删除</button>
        </div>
        <h3>{{ session.title || '新会话' }}</h3>
        <p>{{ sessionSummary(session) }}</p>
        <dl>
          <div>
            <dt>创建时间</dt>
            <dd>{{ formatSessionTime(session.created_at) }}</dd>
          </div>
          <div>
            <dt>更新时间</dt>
            <dd>{{ formatSessionTime(session.updated_at || session.created_at) }}</dd>
          </div>
        </dl>
        <button class="enterprise-button" @click="selectSession(session)">继续对话</button>
      </article>
    </section>

    <div v-if="showNewSessionDialog" class="session-modal-backdrop" @click.self="showNewSessionDialog = false">
      <section class="session-modal enterprise-card">
        <h3>新会话</h3>
        <p>输入第一个问题，系统会创建会话并进入问答页。</p>
        <textarea v-model="newSessionQuery" class="enterprise-textarea" maxlength="200" placeholder="请输入您的问题..."></textarea>
        <div class="modal-actions">
          <button class="enterprise-button secondary" @click="showNewSessionDialog = false">取消</button>
          <button class="enterprise-button" :disabled="!newSessionQuery.trim()" @click="confirmNewSession">开始对话</button>
        </div>
      </section>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { showToast } from 'vant';
import { useSessionStore } from '../store/session';
import { useUserStore } from '../store/user';

const router = useRouter();
const route = useRoute();
const sessionStore = useSessionStore();
const userStore = useUserStore();

const showNewSessionDialog = ref(false);
const newSessionQuery = ref('');
const searchKeyword = ref('');
const activeFilter = ref('all');
const filters = [
  { label: '全部', value: 'all' },
  { label: '最近 7 天', value: 'recent' }
];

const RECENT_WINDOW_MS = 7 * 24 * 60 * 60 * 1000;

const isRecentSession = (session, now) => {
  const time = new Date(session.updated_at || session.created_at).getTime();
  return Number.isFinite(time) && time <= now && now - time <= RECENT_WINDOW_MS;
};

const filteredSessions = computed(() => {
  const keyword = searchKeyword.value.trim().toLowerCase();
  const now = Date.now();
  return sessionStore.sessions.filter((session) => {
    const titleMatched = !keyword || (session.title || '新会话').toLowerCase().includes(keyword);
    const filterMatched = activeFilter.value === 'all'
      || (activeFilter.value === 'recent' && isRecentSession(session, now));
    return titleMatched && filterMatched;
  });
});

const sessionSummary = (session) => session.summary || session.title || '该会话暂无摘要，点击继续对话查看详情。';
const sessionStatus = (session) => session.session_id === sessionStore.currentSession?.session_id ? '当前会话' : '历史会话';

// 监听路由变化，确保每次访问会话管理页面时自动刷新会话列表
watch(() => route.path, async (newPath) => {
  if (newPath === '/sessions') {
    await loadSessions();
  }
});

// 加载会话列表
const loadSessions = async () => {
  // 检查是否登录
  if (!userStore.getLoginStatus) {
    showToast('请先登录');
    router.push('/login');
    return;
  }

  // 获取用户ID（假设从用户信息中获取）
  if (!userStore.userInfo) {
    const result = await userStore.getUserInfoDetail();
    if (!result.success) {
      showToast('获取用户信息失败');
      return;
    }
  }

  if (userStore.userInfo) {


    // 尝试获取用户ID，支持不同的字段名
    let userId = userStore.userInfo.uuid || userStore.userInfo.id || userStore.userInfo.user_id;

    if (userId) {
      await sessionStore.getUserSessions(userId);
    } else {
      // 显示详细的错误信息
      showToast('获取用户ID失败，请检查用户信息结构');
      console.error('用户信息中没有找到ID字段:', userStore.userInfo);
    }
  } else {
    showToast('获取用户信息失败');
  }
};

// 组件挂载时获取会话列表
onMounted(async () => {
  await loadSessions();
});

// 格式化会话时间
const formatSessionTime = (timeString) => {
  if (!timeString) return '';
  try {
    const date = new Date(timeString);
    return date.toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    });
  } catch (error) {
    return timeString;
  }
};

// 选择会话
const selectSession = (session) => {
  // 跳转到带会话ID的路由
  router.push(`/aichat/${session.session_id}`);
};

// 删除会话
const deleteSession = async (sessionId) => {


  const result = await sessionStore.deleteSession(sessionId);
  if (result.success) {
    showToast('会话删除成功');
  } else {
    showToast(result.message || '删除失败');
  }
};

// 打开新会话对话框
const createNewSession = () => {
  showNewSessionDialog.value = true;
};

// 确认创建新会话
const confirmNewSession = async () => {
  if (!newSessionQuery.value.trim()) return;

  // 显示加载状态，保存返回的toast实例
  const toastInstance = showToast({
    type: 'loading',
    message: '创建会话中...',
    forbidClick: true,
    duration: 0
  });

  try {
    const result = await sessionStore.createSession(newSessionQuery.value);
    if (result.success && result.data?.session_id) {
      showToast('会话创建成功');
      showNewSessionDialog.value = false;
      newSessionQuery.value = '';
      // 跳转到带会话ID的聊天页面
      router.push(`/aichat/${result.data.session_id}`);
    } else {
      showToast(result.message || '创建会话失败');
    }
  } catch (error) {
    showToast('创建会话失败');
    console.error('创建会话失败:', error);
  } finally {
    // 使用toast实例的关闭方法
    if (toastInstance && toastInstance.close) {
      toastInstance.close();
    }
  }
};
</script>

<style scoped>
.sessions-page {
  display: grid;
  gap: 20px;
  min-height: 100%;
}

.sessions-hero,
.sessions-toolbar,
.sessions-state {
  padding: 24px;
}

.sessions-hero {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  overflow: hidden;
  position: relative;
  background:
    radial-gradient(circle at 12% 18%, rgba(96, 165, 250, 0.28), transparent 28%),
    linear-gradient(135deg, rgba(30, 58, 138, 0.96), rgba(37, 99, 235, 0.9));
  color: #ffffff;
  border: 1px solid rgba(255, 255, 255, 0.22);
}

.sessions-hero::after {
  content: '';
  position: absolute;
  inset: auto -52px -72px auto;
  width: 220px;
  height: 220px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.12);
  pointer-events: none;
}

.sessions-hero h2 {
  margin: 8px 0;
  font-size: clamp(28px, 4vw, 42px);
  line-height: 1;
  letter-spacing: -0.04em;
}

.sessions-hero p {
  margin: 0;
  color: rgba(255, 255, 255, 0.78);
}

.sessions-toolbar {
  display: grid;
  grid-template-columns: minmax(260px, 1fr) auto;
  align-items: center;
  gap: 16px;
}

.session-filters {
  display: flex;
  gap: 10px;
  justify-content: flex-end;
}

.session-filters button {
  border: 1px solid rgba(37, 99, 235, 0.18);
  border-radius: 999px;
  padding: 9px 16px;
  color: #1e3a8a;
  background: rgba(37, 99, 235, 0.06);
  font-weight: 700;
  cursor: pointer;
  transition: all 0.2s ease;
}

.session-filters button:hover,
.session-filters button.active {
  color: #ffffff;
  border-color: #2563eb;
  background: #2563eb;
  box-shadow: 0 14px 26px rgba(37, 99, 235, 0.22);
}

.sessions-state {
  display: grid;
  place-items: center;
  min-height: 280px;
  text-align: center;
}

.sessions-state h3 {
  margin: 0;
  color: #0f172a;
  font-size: 24px;
}

.sessions-state p {
  margin: 12px 0 20px;
  color: #64748b;
}

.sessions-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 18px;
}

.session-card {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 22px;
  min-height: 280px;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.session-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 22px 48px rgba(15, 23, 42, 0.12);
}

.session-card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.session-card h3 {
  margin: 0;
  color: #0f172a;
  font-size: 22px;
  line-height: 1.2;
}

.session-card p {
  margin: 0;
  color: #64748b;
  line-height: 1.7;
  flex: 1;
}

.session-card dl {
  display: grid;
  gap: 10px;
  margin: 0;
  padding: 14px 0;
  border-top: 1px solid rgba(148, 163, 184, 0.22);
  border-bottom: 1px solid rgba(148, 163, 184, 0.22);
}

.session-card dl div {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

.session-card dt {
  color: #94a3b8;
  font-size: 12px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.session-card dd {
  margin: 0;
  color: #334155;
  font-size: 13px;
  font-weight: 700;
}

.enterprise-button.danger {
  background: #dc2626;
  box-shadow: 0 12px 22px rgba(220, 38, 38, 0.2);
}

.enterprise-button.secondary {
  color: #1e3a8a;
  background: rgba(37, 99, 235, 0.08);
  box-shadow: none;
}

.enterprise-button:disabled {
  cursor: not-allowed;
  opacity: 0.48;
  box-shadow: none;
}

.session-modal-backdrop {
  position: fixed;
  inset: 0;
  z-index: 1000;
  display: grid;
  place-items: center;
  padding: 24px;
  background: rgba(15, 23, 42, 0.56);
  backdrop-filter: blur(10px);
}

.session-modal {
  width: min(520px, 100%);
  padding: 28px;
  box-shadow: 0 32px 80px rgba(15, 23, 42, 0.28);
}

.session-modal h3 {
  margin: 0 0 8px;
  color: #0f172a;
  font-size: 26px;
}

.session-modal p {
  margin: 0 0 18px;
  color: #64748b;
}

.enterprise-textarea {
  width: 100%;
  min-height: 132px;
  resize: vertical;
  border: 1px solid rgba(148, 163, 184, 0.32);
  border-radius: 18px;
  padding: 14px 16px;
  color: #0f172a;
  background: rgba(248, 250, 252, 0.86);
  box-sizing: border-box;
  font: inherit;
  outline: none;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.enterprise-textarea:focus {
  border-color: #2563eb;
  box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.12);
}

.modal-actions {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  margin-top: 18px;
}

@media (max-width: 720px) {
  .sessions-hero {
    align-items: flex-start;
    flex-direction: column;
  }

  .sessions-toolbar {
    grid-template-columns: 1fr;
  }

  .session-filters {
    justify-content: flex-start;
    flex-wrap: wrap;
  }

  .sessions-grid {
    grid-template-columns: minmax(0, 1fr);
  }

  .modal-actions {
    flex-direction: column-reverse;
  }
}
</style>
