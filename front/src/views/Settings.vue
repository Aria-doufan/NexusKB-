<template>
  <div class="settings-page">
    <aside class="settings-sidebar enterprise-card">
      <button v-for="section in sections" :key="section.id" :class="{ active: activeSection === section.id }" @click="activeSection = section.id">
        <strong>{{ section.label }}</strong>
        <span>{{ section.description }}</span>
      </button>
    </aside>

    <section class="settings-panel enterprise-card">
      <p class="shell-eyebrow">Settings</p>
      <h2>{{ activeSectionMeta.label }}</h2>
      <p>{{ activeSectionMeta.description }}</p>

      <div v-if="activeSection === 'display'" class="settings-list">
        <article v-for="theme in themeList" :key="theme.id" class="setting-row" :class="{ active: currentTheme === theme.id }">
          <span class="theme-swatch" :style="{ backgroundColor: theme.primaryColor }"></span>
          <div>
            <strong>{{ theme.name }}</strong>
            <span>主题标识：{{ theme.id }}</span>
          </div>
          <button class="enterprise-button secondary" @click="changeTheme(theme.id)">应用</button>
        </article>
      </div>

      <div v-else-if="activeSection === 'language'" class="settings-list">
        <article v-for="lang in languageOptions" :key="lang.value" class="setting-row" :class="{ active: currentLanguage === lang.value }">
          <div>
            <strong>{{ lang.label }}</strong>
            <span>{{ lang.value }}</span>
          </div>
          <button class="enterprise-button secondary" @click="selectLanguage(lang.value)">选择</button>
        </article>
        <button class="enterprise-button" @click="changeLanguage">保存语言设置</button>
      </div>

      <div v-else class="settings-list">
        <article class="setting-row">
          <div>
            <strong>{{ activeSectionMeta.emptyTitle }}</strong>
            <span>{{ activeSectionMeta.emptyDescription }}</span>
          </div>
          <span class="enterprise-tag">规划中</span>
        </article>
      </div>
    </section>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue';
import { showToast } from 'vant';
import { useThemeStore } from '../store/theme';
import { useI18n } from 'vue-i18n';
import { useLanguageStore } from '../store/language';

const themeStore = useThemeStore();
const languageStore = useLanguageStore();
const { t, locale } = useI18n();

const activeSection = ref('display');
const sections = [
  { id: 'general', label: '通用', description: '基础使用偏好', emptyTitle: '通用设置', emptyDescription: '当前版本暂无额外通用设置。' },
  { id: 'display', label: '显示', description: '主题与界面风格', emptyTitle: '显示设置', emptyDescription: '选择一个主题并立即应用。' },
  { id: 'language', label: '语言', description: '中英文界面切换', emptyTitle: '语言设置', emptyDescription: '选择语言后保存并刷新页面。' },
  { id: 'model', label: '模型', description: '默认模型和接口状态', emptyTitle: '模型设置', emptyDescription: '模型与接口状态展示将在后端提供配置接口后接入。' },
  { id: 'privacy', label: '隐私', description: '账户与数据偏好', emptyTitle: '隐私设置', emptyDescription: '隐私偏好将在账户策略完善后接入。' }
];
const activeSectionMeta = computed(() => sections.find((section) => section.id === activeSection.value) || sections[0]);
const selectLanguage = (language) => {
  currentLanguage.value = language;
};

// 主题相关
const themeList = computed(() => themeStore.getAllThemes);
const currentTheme = computed(() => themeStore.getCurrentTheme);

// 切换主题
const changeTheme = (themeId) => {
  themeStore.setTheme(themeId);
  showToast(t('settings.themeChanged'));
};

// 语言相关
const currentLanguage = ref(languageStore.getCurrentLanguage);
const languageOptions = [
  { label: '简体中文', value: 'zh-CN' },
  { label: 'English', value: 'en-US' }
];

// 切换语言
const changeLanguage = () => {
  languageStore.setLanguage(currentLanguage.value);
  locale.value = currentLanguage.value;
  showToast(t('settings.languageChanged'));
  // 强制刷新页面以应用语言更改
  window.location.reload();
};
</script>

<style scoped>
.settings-page {
  display: grid;
  grid-template-columns: 300px 1fr;
  gap: 20px;
  min-height: 100%;
}

.settings-sidebar,
.settings-panel {
  padding: 20px;
}

.settings-sidebar {
  display: flex;
  flex-direction: column;
  gap: 10px;
  align-self: start;
}

.settings-sidebar button {
  display: flex;
  flex-direction: column;
  gap: 6px;
  width: 100%;
  border: 1px solid transparent;
  border-radius: 14px;
  background: transparent;
  color: var(--color-text);
  padding: 14px 16px;
  text-align: left;
  cursor: pointer;
  transition: background-color 0.2s ease, border-color 0.2s ease, color 0.2s ease;
}

.settings-sidebar button:hover,
.settings-sidebar button.active {
  border-color: rgba(37, 99, 235, 0.16);
  background: rgba(37, 99, 235, 0.08);
  color: var(--color-primary);
}

.settings-sidebar strong {
  font-size: 15px;
}

.settings-sidebar span,
.settings-panel > p,
.setting-row span {
  color: var(--color-muted);
  font-size: 13px;
}

.shell-eyebrow {
  margin: 0 0 8px;
  color: var(--color-primary);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.settings-panel h2 {
  margin: 0 0 8px;
  color: var(--color-text);
  font-size: 28px;
}

.settings-panel > p:not(.shell-eyebrow) {
  margin: 0;
}

.settings-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-top: 22px;
}

.setting-row {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  align-items: center;
  gap: 14px;
  border: 1px solid var(--color-border);
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.72);
  padding: 14px;
}

.setting-row.active {
  border-color: rgba(37, 99, 235, 0.32);
  background: rgba(37, 99, 235, 0.08);
}

.setting-row > div {
  display: flex;
  flex-direction: column;
  gap: 5px;
  min-width: 0;
}

.setting-row strong {
  color: var(--color-text);
  font-size: 15px;
}

.theme-swatch {
  width: 34px;
  height: 34px;
  border: 2px solid rgba(255, 255, 255, 0.85);
  border-radius: 999px;
  box-shadow: 0 0 0 1px var(--color-border);
}

.enterprise-tag {
  justify-self: start;
}

@media (max-width: 820px) {
  .settings-page {
    grid-template-columns: 1fr;
  }

  .setting-row {
    grid-template-columns: 1fr;
    align-items: start;
  }

  .theme-swatch {
    width: 42px;
    height: 42px;
  }
}
</style>
