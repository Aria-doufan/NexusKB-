<template>
  <router-view v-slot="{ Component }">
    <component v-if="isAuthRoute" :is="Component" />
    <AppShell v-else>
      <template v-if="$route.meta.keepAlive">
        <keep-alive>
          <component :is="Component" />
        </keep-alive>
      </template>
      <template v-else>
        <component :is="Component" />
      </template>
    </AppShell>
  </router-view>
</template>

<script setup>
import { computed } from 'vue';
import { useRoute } from 'vue-router';
import AppShell from './components/AppShell.vue';

const route = useRoute();
const isAuthRoute = computed(() => route.path === '/login' || route.path === '/register');
</script>
