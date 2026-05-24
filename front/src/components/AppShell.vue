<template>
  <div class="app-shell">
    <aside class="shell-sidebar">
      <RouterLink to="/aichat" class="shell-brand" aria-label="NexusKB 首页">
        <span class="brand-mark">N</span>
        <span>
          <strong>NexusKB</strong>
          <small>企业知识库</small>
        </span>
      </RouterLink>

      <nav class="shell-nav" aria-label="主导航">
        <RouterLink v-for="item in navItems" :key="item.to" :to="item.to" class="shell-nav-item">
          <span class="nav-icon">{{ item.icon }}</span>
          <span>{{ item.label }}</span>
        </RouterLink>
      </nav>

      <div class="shell-sidebar-footer">
        <div class="shell-status-dot"></div>
        <div>
          <strong>知识库在线</strong>
          <span>RAG 服务就绪</span>
        </div>
      </div>
    </aside>

    <section class="shell-main">
      <header class="shell-toolbar">
        <div>
          <p class="shell-eyebrow">Enterprise Knowledge Cockpit</p>
          <h1>{{ pageTitle }}</h1>
        </div>
        <div class="shell-toolbar-actions">
          <span class="shell-chip">{{ languageLabel }}</span>
          <span class="shell-chip">{{ themeLabel }}</span>
          <RouterLink to="/my" class="shell-user">
            <span class="shell-avatar">{{ userInitial }}</span>
            <span>{{ userName }}</span>
          </RouterLink>
        </div>
      </header>

      <main class="shell-content">
        <slot />
      </main>
    </section>
  </div>
</template>

<script setup>
import { computed } from 'vue';
import { useRoute, RouterLink } from 'vue-router';
import { useUserStore } from '../store/user';
import { useThemeStore } from '../store/theme';
import { useLanguageStore } from '../store/language';

const route = useRoute();
const userStore = useUserStore();
const themeStore = useThemeStore();
const languageStore = useLanguageStore();

const navItems = [
  { to: '/aichat', label: 'AI 问答', icon: '💬' },
  { to: '/sessions', label: '会话管理', icon: '🗂' },
  { to: '/my', label: '个人中心', icon: '👤' },
  { to: '/settings', label: '设置', icon: '⚙' },
];

const pageTitle = computed(() => route.meta.title || 'NexusKB');
const userName = computed(() => userStore.userInfo?.username || '未登录用户');
const userInitial = computed(() => userName.value.slice(0, 1).toUpperCase());
const themeLabel = computed(() => themeStore.getAllThemes.find((theme) => theme.id === themeStore.getCurrentTheme)?.name || '浅色模式');
const languageLabel = computed(() => languageStore.getCurrentLanguage === 'en-US' ? 'English' : '简体中文');
</script>

<style scoped>
.app-shell {
  min-height: 100vh;
  display: grid;
  grid-template-columns: 248px minmax(0, 1fr);
  background: var(--color-bg);
  color: var(--color-text);
}

.shell-sidebar {
  position: sticky;
  top: 0;
  height: 100vh;
  display: flex;
  flex-direction: column;
  padding: 24px 18px;
  background: linear-gradient(180deg, var(--color-shell) 0%, #172554 100%);
  color: #fff;
}

.shell-brand,
.shell-nav-item,
.shell-user {
  color: inherit;
  text-decoration: none;
}

.shell-brand {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 4px 6px 28px;
}

.brand-mark {
  width: 42px;
  height: 42px;
  display: grid;
  place-items: center;
  border-radius: 14px;
  background: #fff;
  color: var(--color-primary);
  font-weight: 800;
}

.shell-brand strong,
.shell-brand small,
.shell-sidebar-footer strong,
.shell-sidebar-footer span {
  display: block;
}

.shell-brand small,
.shell-sidebar-footer span {
  color: rgba(255, 255, 255, 0.68);
  font-size: 12px;
}

.shell-nav {
  display: grid;
  gap: 8px;
}

.shell-nav-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 14px;
  border-radius: 14px;
  color: rgba(255, 255, 255, 0.78);
}

.shell-nav-item.router-link-active {
  background: rgba(255, 255, 255, 0.14);
  color: #fff;
  box-shadow: inset 3px 0 0 #93c5fd;
}

.nav-icon {
  width: 22px;
  text-align: center;
}

.shell-sidebar-footer {
  margin-top: auto;
  display: flex;
  gap: 10px;
  align-items: center;
  padding: 14px;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.12);
}

.shell-status-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  background: #22c55e;
  box-shadow: 0 0 0 6px rgba(34, 197, 94, 0.14);
}

.shell-main {
  min-width: 0;
  display: flex;
  flex-direction: column;
}

.shell-toolbar {
  height: 86px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  padding: 0 32px;
  background: rgba(255, 255, 255, 0.82);
  border-bottom: 1px solid var(--color-border);
  backdrop-filter: blur(16px);
}

.shell-eyebrow {
  margin: 0 0 4px;
  color: var(--color-muted);
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.shell-toolbar h1 {
  margin: 0;
  font-size: 24px;
  line-height: 1.2;
}

.shell-toolbar-actions {
  display: flex;
  align-items: center;
  gap: 10px;
}

.shell-chip,
.shell-user {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-height: 36px;
  padding: 0 12px;
  border-radius: 999px;
  background: var(--color-card);
  border: 1px solid var(--color-border);
  color: var(--color-text);
  font-size: 13px;
}

.shell-avatar {
  width: 24px;
  height: 24px;
  display: grid;
  place-items: center;
  border-radius: 999px;
  background: var(--color-primary-soft);
  color: var(--color-primary);
  font-weight: 700;
}

.shell-content {
  min-width: 0;
  flex: 1;
  padding: 24px 32px 32px;
}

@media (max-width: 960px) {
  .app-shell {
    grid-template-columns: 82px minmax(0, 1fr);
  }

  .shell-sidebar {
    padding: 18px 12px;
  }

  .shell-brand span:last-child,
  .shell-nav-item span:last-child,
  .shell-sidebar-footer div {
    display: none;
  }

  .shell-toolbar {
    padding: 0 20px;
  }
}
</style>
