import { createRouter, createWebHistory } from 'vue-router';
import HomePage from '../views/HomePage.vue';
import LoginPage from '../views/LoginPage.vue';
import RegisterPage from '../views/RegisterPage.vue';

const routes = [
  {
    path: '/',
    name: 'HomePage',
    component: HomePage,
    meta: { requiresAuth: true } // 首页需要登录
  },
  {
    path: '/login',
    name: 'LoginPage',
    component: LoginPage,
    meta: { public: true } // 公开页面
  },
  {
    path: '/register',
    name: 'RegisterPage',
    component: RegisterPage,
    meta: { public: true } // 公开页面
  }
];

const router = createRouter({
  history: createWebHistory(),
  routes
});

// 路由守卫
router.beforeEach((to, from, next) => {
  const isLoggedIn = !!localStorage.getItem('token');

  // 调试信息，检查守卫是否触发及关键变量的值
  console.log('Navigating to:', to.path);
  console.log('Requires Auth:', to.meta.requiresAuth);
  console.log('Is Public:', to.meta.public);
  console.log('Is Logged In:', isLoggedIn);

  if (to.meta.requiresAuth && !isLoggedIn) {
    // 未登录用户访问需要授权的页面，重定向到登录页
    next('/login');
  } else if (to.meta.public && isLoggedIn && to.path !== '/login') {
    // 已登录用户访问公开页面（除了 /login），重定向到首页
    // 允许访问 /login 以便注销或切换账号
    next('/');
  } else {
    // 其他情况正常跳转
    next();
  }
});

export default router;