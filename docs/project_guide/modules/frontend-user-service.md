# 前端与用户服务模块设计

整理日期：2026-05-15

## 职责

本模块记录 `front` 与 `DjangoUserService` 的边界。

- `front`：Vue 前端，负责登录注册、聊天 UI、会话列表、个人信息和设置页面。
- `DjangoUserService`：用户注册、登录、Token、用户资料、文件上传等用户侧服务。
- `backend`：Chat Agent 与 RAG 主服务。

## 当前接口来源

用户服务 API 的当前维护入口是：

- [Django 用户服务 API](../../../DjangoUserService/api.md)

`front/api.md` 与该文档内容重复，后续维护时以 `DjangoUserService/api.md` 为准。

## 前端核心目录

| 目录或文件 | 说明 |
| --- | --- |
| `front/src/views/AIChat.vue` | 聊天主界面 |
| `front/src/views/Login.vue` | 登录 |
| `front/src/views/Register.vue` | 注册 |
| `front/src/views/Sessions.vue` | 会话列表 |
| `front/src/store/` | 用户、会话、主题、语言等状态 |
| `front/src/config/api.js` | API 地址配置 |
| `front/src/router/index.js` | 前端路由 |

## 当前不足

- 前端缺少单独的交互流程文档。
- 登录态、Token 刷新、接口错误处理的约定还应进一步写清楚。
- Chat SSE、Router 非流式接口、普通 REST 接口之间的前端调用策略需要整理。

## 下一步

- 补一份前端状态流转图：登录、进入聊天、切换会话、发送消息、接收流式响应。
- 明确前端到底调用 `/api/agent/query/stream` 还是 Router API，避免入口分叉。
- 给用户服务 API 增加认证失败、Token 过期、刷新失败的前端处理约定。
