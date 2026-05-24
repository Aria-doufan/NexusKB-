<template>
  <div class="account-page">
    <section class="account-overview-card enterprise-card">
      <div class="account-avatar">{{ userInitial }}</div>
      <div>
        <p class="shell-eyebrow">Account Overview</p>
        <h2>{{ isLogin && userInfo ? userInfo.username : $t('my.notLoggedIn') }}</h2>
        <p>{{ isLogin && userInfo ? userBio : '登录后可查看个人资料和历史会话。' }}</p>
      </div>
      <div class="account-actions">
        <button v-if="isLogin" class="enterprise-button" @click="goToProfile">编辑资料</button>
        <button v-if="!isLogin" class="enterprise-button" @click="goToLogin">{{ $t('my.goToLogin') }}</button>
        <button v-if="!isLogin" class="enterprise-button secondary" @click="goToRegister">{{ $t('my.goToRegister') }}</button>
      </div>
    </section>

    <section class="account-grid">
      <article class="enterprise-card account-stat">
        <span class="enterprise-tag">会话</span>
        <strong>历史会话</strong>
        <p>继续查看和管理企业知识问答记录。</p>
        <button class="enterprise-button secondary" @click="$router.push('/sessions')">进入会话管理</button>
      </article>

      <article class="enterprise-card account-stat">
        <span class="enterprise-tag">设置</span>
        <strong>偏好设置</strong>
        <p>调整语言、主题和知识库使用偏好。</p>
        <button class="enterprise-button secondary" @click="goToSettings">进入设置</button>
      </article>

      <article v-if="isLogin" class="enterprise-card account-stat danger-zone">
        <span class="enterprise-tag">账户</span>
        <strong>{{ $t('my.logout') }}</strong>
        <p>退出当前账号并清除本地登录状态。</p>
        <button class="enterprise-button danger" @click="handleLogout">{{ $t('my.logout') }}</button>
      </article>
    </section>
  </div>
</template>

<script setup>
import { onMounted } from 'vue';
import { useUserStore } from '../store/user';
import { useRouter } from 'vue-router';
import { computed, ref } from 'vue';
import { showDialog, showToast } from 'vant';
import { useI18n } from 'vue-i18n';

const userStore = useUserStore();
const router = useRouter();
const { t } = useI18n();

// 从store获取用户信息和登录状态
const userInfo = computed(() => userStore.userInfo);
const isLogin = computed(() => userStore.getLoginStatus);
const userBio = computed(() => userStore.getUserBio || t('profile.bio'));
const userInitial = computed(() => (userInfo.value?.username || 'N').slice(0, 1).toUpperCase());

// 跳转到登录页
const goToLogin = () => {
  router.push('/login');
};

// 跳转到注册页
const goToRegister = () => {
  router.push('/register');
};

// 跳转到个人信息页
const goToProfile = () => {
  if (isLogin.value) {
    router.push('/profile');
  }
};



// 跳转到设置页面
const goToSettings = () => {
  router.push('/settings');
};

// 退出登录
const handleLogout = () => {
  showDialog({
    title: t('common.confirm'),
    message: t('my.logout') + '?',
    showCancelButton: true,
  }).then((action) => {
    if (action === 'confirm') {
      userStore.logout();
      router.push('/login');
    }
  });
};

// 获取用户信息
onMounted(async () => {
  try {
    await userStore.getUserInfoDetail();
  } catch (error) {
    console.error('获取用户信息失败:', error);
  }
});
</script>

<style scoped>
.account-page {
  display: grid;
  gap: 20px;
}

.account-overview-card {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  align-items: center;
  gap: 24px;
  padding: 28px;
  background:
    radial-gradient(circle at top left, rgba(37, 99, 235, 0.18), transparent 34%),
    linear-gradient(135deg, rgba(255, 255, 255, 0.98), rgba(239, 246, 255, 0.92));
}

.account-overview-card h2 {
  margin: 6px 0 8px;
  color: var(--color-text);
  font-size: clamp(1.6rem, 3vw, 2.35rem);
  line-height: 1.1;
}

.account-overview-card p:not(.shell-eyebrow) {
  margin: 0;
  color: var(--color-muted);
}

.account-avatar {
  width: 86px;
  height: 86px;
  border-radius: 28px;
  display: grid;
  place-items: center;
  color: #fff;
  font-size: 2.2rem;
  font-weight: 800;
  letter-spacing: 0.04em;
  background: linear-gradient(135deg, #1d4ed8, #38bdf8);
  box-shadow: 0 18px 35px rgba(37, 99, 235, 0.28);
}

.account-actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 10px;
}

.account-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 20px;
}

.account-stat {
  display: grid;
  align-content: start;
  gap: 14px;
  padding: 24px;
}

.account-stat strong {
  color: var(--color-text);
  font-size: 1.25rem;
}

.account-stat p {
  margin: 0;
  color: var(--color-muted);
  line-height: 1.7;
}

.danger-zone {
  border-color: rgba(220, 38, 38, 0.18);
}

.enterprise-button.danger {
  background: linear-gradient(135deg, #dc2626, #f97316);
  box-shadow: 0 14px 26px rgba(220, 38, 38, 0.22);
}

.shell-eyebrow {
  margin: 0;
  color: var(--color-primary);
  font-size: 0.75rem;
  font-weight: 800;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}

@media (max-width: 760px) {
  .account-overview-card {
    grid-template-columns: 1fr;
    align-items: start;
  }

  .account-actions {
    justify-content: flex-start;
  }
}
</style>