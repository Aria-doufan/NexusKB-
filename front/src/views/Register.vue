<template>
  <div class="auth-page">
    <section class="auth-brand-panel">
      <RouterLink to="/aichat" class="auth-brand">NexusKB</RouterLink>
      <h1>加入企业知识问答平台</h1>
      <p>注册后即可创建知识库问答会话，保留历史上下文，并在回答中查看可追溯依据。</p>
      <div class="auth-capabilities">
        <span>可靠检索</span>
        <span>多轮记忆</span>
        <span>引用溯源</span>
      </div>
    </section>

    <section class="auth-card enterprise-card">
      <p class="shell-eyebrow">Create account</p>
      <h2>注册 NexusKB</h2>
      <div class="auth-form">
        <van-field v-model="form.username" placeholder="请输入用户名" :rules="usernameRules" required left-icon="user-o" @blur="validateUsername" />
        <van-field v-model="form.email" placeholder="请输入邮箱地址" :rules="emailRules" required type="email" left-icon="envelop-o" @blur="validateEmail" />
        <van-field v-model="form.telephone" placeholder="请输入手机号码" type="tel" left-icon="phone" maxlength="11" />
        <van-field v-model="form.password" placeholder="请输入密码（6-20位）" :rules="passwordRules" required type="password" left-icon="lock" @blur="validatePassword" />
        <van-field v-model="form.confirm_password" placeholder="请确认密码" :rules="confirmPasswordRules" required type="password" left-icon="lock" @blur="validateConfirmPassword" />
        <button class="enterprise-button auth-submit" :disabled="loading" @click="handleRegister">
          {{ loading ? '注册中...' : '注册' }}
        </button>
      </div>
      <p class="auth-switch">已有账号？<button @click="goToLogin">去登录</button></p>
    </section>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue';
import { useRouter, RouterLink } from 'vue-router';
import { showToast, showDialog } from 'vant';
import { useUserStore } from '../store/user';

const router = useRouter();
const userStore = useUserStore();

const loading = ref(false);

const form = reactive({
  username: '',
  email: '',
  telephone: '',
  password: '',
  confirm_password: ''
});

const usernameRules = [
  { required: true, message: '请输入用户名' }
];

const emailRules = [
  { required: true, message: '请输入邮箱地址' },
  { pattern: /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/, message: '请输入正确的邮箱地址' }
];

const passwordRules = [
  { required: true, message: '请输入密码' },
  { pattern: /^.{6,20}$/, message: '密码长度应为6-20位' }
];

const confirmPasswordRules = [
  { required: true, message: '请确认密码' }
];

const validateUsername = () => {
  if (!form.username) {
    showToast('请输入用户名');
    return false;
  }
  return true;
};

const validateEmail = () => {
  if (!form.email) {
    showToast('请输入邮箱地址');
    return false;
  }
  const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
  if (!emailRegex.test(form.email)) {
    showToast('请输入正确的邮箱地址');
    return false;
  }
  return true;
};

const validatePassword = () => {
  if (!form.password) {
    showToast('请输入密码');
    return false;
  }
  if (form.password.length < 6 || form.password.length > 20) {
    showToast('密码长度应为6-20位');
    return false;
  }
  return true;
};

const validateConfirmPassword = () => {
  if (!form.confirm_password) {
    showToast('请确认密码');
    return false;
  }
  if (form.password !== form.confirm_password) {
    showToast('两次输入的密码不一致');
    return false;
  }
  return true;
};

const validateForm = () => {
  return validateUsername() && validateEmail() && validatePassword() && validateConfirmPassword();
};

const handleRegister = async () => {
  console.log('handleRegister函数被调用');
  console.log('表单数据:', form);

  if (!validateForm()) {
    console.log('表单验证失败');
    return;
  }

  console.log('表单验证通过，开始注册');
  loading.value = true;

  try {
    console.log('调用userStore.register方法');
    const result = await userStore.register(form);

    console.log('注册结果:', result);

    if (result.success) {
      showToast({
        message: result.message,
        position: 'top'
      });

      // 注册成功后跳转到对话页面
      setTimeout(() => {
        router.push('/aichat');
      }, 1500);
    } else {
      showToast({
        message: result.message,
        position: 'top',
        type: 'fail'
      });
    }
  } catch (error) {
    console.error('注册失败:', error);
    showToast({
      message: '注册失败，请稍后重试',
      position: 'top',
      type: 'fail'
    });
  } finally {
    loading.value = false;
  }
};

const goToLogin = () => {
  router.push('/login');
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
