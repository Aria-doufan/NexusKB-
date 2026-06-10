<template>
  <div class="auth-page">
    <section class="auth-brand-panel">
      <RouterLink to="/aichat" class="auth-brand">NexusKB</RouterLink>
      <h1>企业知识问答驾驶舱</h1>
      <p>通过可靠检索、多轮记忆和引用溯源，把内部文档转化为可信答案。</p>
      <div class="auth-capabilities">
        <span>可靠检索</span>
        <span>多轮记忆</span>
        <span>引用溯源</span>
      </div>
    </section>

    <section class="auth-card enterprise-card">
      <p class="shell-eyebrow">Welcome back</p>
      <h2>登录 NexusKB</h2>
      <van-form @submit="onSubmit" class="auth-form">
        <van-field v-model="username" name="username" label="用户名" placeholder="请输入用户名" :rules="[{ required: true, message: '请填写用户名' }]" />
        <van-field v-model="password" type="password" name="password" label="密码" placeholder="请输入密码" :rules="[{ required: true, message: '请填写密码' }]" />
        <button class="enterprise-button auth-submit" type="submit">登录</button>
      </van-form>
      <p class="auth-switch">还没有账号？<button @click="goToRegister">去注册</button></p>
    </section>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import { useRouter, RouterLink } from 'vue-router';
import { showToast } from 'vant';
import { useUserStore } from '../store/user';

const router = useRouter();
const userStore = useUserStore();

const username = ref('');
const password = ref('');

const onSubmit = async (values) => {
  // 显示加载提示
  showToast({
    type: 'loading',
    message: '登录中...',
    forbidClick: true,
    duration: 0
  });

  try {
    // 调用API登录
    const result = await userStore.login({
      username: username.value,
      password: password.value
    });

    if (result.success) {
      showToast({
        type: 'success',
        message: result.message
      });

      await router.push('/aichat');
    } else {
      showToast({
        type: 'fail',
        message: result.message
      });
    }
  } catch (error) {
    showToast({
      type: 'fail',
      message: '登录失败，请稍后再试'
    });
  }
};

const goToRegister = () => {
  router.push('/register');
};
</script>

<style scoped>
.auth-page {
  min-height: 100vh;
  display: grid;
  grid-template-columns: minmax(420px, 0.95fr) minmax(380px, 1.05fr);
  background: radial-gradient(circle at 20% 10%, rgba(147, 197, 253, 0.4), transparent 30%), linear-gradient(135deg, #eff6ff 0%, #f8fafc 48%, #ffffff 100%);
}

.auth-brand-panel {
  padding: 72px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  background: linear-gradient(160deg, #1e3a8a 0%, #2563eb 100%);
  color: #fff;
}

.auth-brand {
  color: #fff;
  text-decoration: none;
  font-size: 22px;
  font-weight: 800;
}

.auth-brand-panel h1 {
  margin: 36px 0 18px;
  font-size: clamp(36px, 5vw, 58px);
  line-height: 1.05;
}

.auth-brand-panel p {
  margin: 0;
  max-width: 560px;
  color: rgba(255, 255, 255, 0.78);
  font-size: 18px;
}

.auth-capabilities {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 32px;
}

.auth-capabilities span {
  padding: 10px 14px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.14);
}

.auth-card {
  align-self: center;
  justify-self: center;
  width: min(460px, calc(100% - 48px));
  padding: 34px;
}

.shell-eyebrow {
  margin: 0 0 4px;
  color: var(--color-muted);
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.auth-card h2 {
  margin: 0 0 24px;
  font-size: 30px;
}

.auth-form {
  display: grid;
  gap: 16px;
}

.auth-submit {
  width: 100%;
  margin-top: 4px;
  min-height: 44px;
}

.auth-switch {
  margin: 22px 0 0;
  text-align: center;
  color: var(--color-muted);
}

.auth-switch button {
  border: 0;
  background: transparent;
  color: var(--color-primary);
  cursor: pointer;
}

@media (max-width: 860px) {
  .auth-page {
    grid-template-columns: 1fr;
  }

  .auth-brand-panel {
    padding: 42px 28px;
  }

  .auth-card {
    margin: 32px 0;
  }
}
</style>
