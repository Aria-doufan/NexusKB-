import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const read = (path) => readFileSync(resolve(root, path), 'utf8');

const checks = [
  {
    name: 'package exposes verify:web-ui',
    pass: () => read('package.json').includes('"verify:web-ui"'),
  },
  {
    name: 'App.vue uses AppShell for desktop app routes',
    pass: () => {
      const app = read('src/App.vue');
      return app.includes('AppShell') && app.includes('isAuthRoute');
    },
  },
  {
    name: 'App.vue no longer caps the app at mobile width',
    pass: () => !read('src/App.vue').includes('max-width: 750px'),
  },
  {
    name: 'global CSS defines enterprise design tokens',
    pass: () => {
      const css = read('src/style.css');
      return css.includes('--color-primary: #2563eb') && css.includes('--color-shell: #1e3a8a');
    },
  },
  {
    name: 'AppShell provides desktop navigation and toolbar',
    pass: () => {
      const shell = read('src/components/AppShell.vue');
      return shell.includes('app-shell') && shell.includes('shell-sidebar') && shell.includes('知识库在线');
    },
  },
  {
    name: 'AIChat uses desktop cockpit layout',
    pass: () => {
      const page = read('src/views/AIChat.vue');
      return page.includes('chat-cockpit') && page.includes('conversation-rail') && page.includes('evidence-panel');
    },
  },
  {
    name: 'AIChat evidence panel is closed by default and toggleable',
    pass: () => {
      const page = read('src/views/AIChat.vue');
      return page.includes('const showEvidencePanel = ref(false)')
        && page.includes('v-if="showEvidencePanel"')
        && page.includes('@click="showEvidencePanel = !showEvidencePanel"')
        && !/\.evidence-panel\s*\{\s*display:\s*none;/.test(page);
    },
  },
  {
    name: 'AIChat no longer renders mobile nav bar or tab bar',
    pass: () => {
      const page = read('src/views/AIChat.vue');
      return !page.includes('<van-nav-bar') && !page.includes('<tab-bar');
    },
  },
  {
    name: 'Sessions page uses desktop management layout',
    pass: () => {
      const page = read('src/views/Sessions.vue');
      return page.includes('sessions-page') && page.includes('sessions-grid') && page.includes('session-modal-backdrop');
    },
  },
  {
    name: 'Sessions page no longer renders mobile nav bar or tab bar',
    pass: () => {
      const page = read('src/views/Sessions.vue');
      return !page.includes('<van-nav-bar') && !page.includes('<tab-bar');
    },
  },
  {
    name: 'Auth pages use enterprise split layout',
    pass: () => {
      const login = read('src/views/Login.vue');
      const register = read('src/views/Register.vue');
      return login.includes('auth-page') && login.includes('auth-brand-panel') && register.includes('auth-page') && register.includes('auth-brand-panel');
    },
  },
  {
    name: 'Auth pages no longer render mobile nav bars',
    pass: () => !read('src/views/Login.vue').includes('<van-nav-bar') && !read('src/views/Register.vue').includes('<van-nav-bar'),
  },
  {
    name: 'Login success navigates directly to AI chat',
    pass: () => read('src/views/Login.vue').includes("router.push('/aichat')"),
  },
  {
    name: 'Account pages use desktop cards',
    pass: () => {
      const my = read('src/views/My.vue');
      const profile = read('src/views/Profile.vue');
      return my.includes('account-page') && my.includes('account-overview-card') && profile.includes('profile-page') && profile.includes('profile-grid');
    },
  },
  {
    name: 'Account pages no longer render mobile nav or tabbar',
    pass: () => {
      const my = read('src/views/My.vue');
      const profile = read('src/views/Profile.vue');
      return !my.includes('<van-nav-bar') && !my.includes('<tab-bar') && !profile.includes('<van-nav-bar') && !profile.includes('<tab-bar');
    },
  },
  {
    name: 'Settings page uses desktop panel layout',
    pass: () => {
      const page = read('src/views/Settings.vue');
      return page.includes('settings-page') && page.includes('settings-sidebar') && page.includes('settings-panel');
    },
  },
  {
    name: 'Settings page no longer renders mobile nav or bottom popups',
    pass: () => {
      const page = read('src/views/Settings.vue');
      return !page.includes('<van-nav-bar') && !page.includes('<van-popup');
    },
  },
];

let failed = 0;

for (const check of checks) {
  if (check.pass()) {
    console.log(`PASS ${check.name}`);
  } else {
    failed += 1;
    console.error(`FAIL ${check.name}`);
  }
}

if (failed > 0) {
  process.exit(1);
}
