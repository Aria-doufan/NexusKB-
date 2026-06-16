import assert from 'node:assert/strict';

import {
  applyProgressEvent,
  createAssistantProgressMessage,
  hideAssistantProgress,
  normalizeProgressEvent,
} from '../src/utils/chatProgress.js';

const assistantMessage = createAssistantProgressMessage();
assert.deepEqual(assistantMessage, {
  role: 'assistant',
  content: '',
  progressStatus: '正在分析问题……',
  progressDetail: '',
  isProgressVisible: true,
});

assert.deepEqual(
  normalizeProgressEvent({
    type: 'retrieval_started',
    event: 'retrieval_started',
    stage: 'retrieve',
    message: null,
    data: { attempt_id: 1 },
  }),
  {
    type: 'progress',
    stage: 'retrieval_started',
    message: '',
    data: { attempt_id: 1 },
  },
);

assert.deepEqual(
  normalizeProgressEvent({
    type: 'rag_event',
    event: 'retrieval_started',
    stage: 'retrieve',
    message: null,
    data: { attempt_id: 2 },
  }),
  {
    type: 'progress',
    stage: 'retrieval_started',
    message: '',
    data: { attempt_id: 2 },
  },
);

assert.deepEqual(
  normalizeProgressEvent({
    type: 'progress',
    stage: 'retrieval_finished',
    message: '已找到 5 篇相关文档',
    data: { selected_documents: 5 },
  }),
  {
    type: 'progress',
    stage: 'retrieval_finished',
    message: '已找到 5 篇相关文档',
    data: { selected_documents: 5 },
  },
);

assert.deepEqual(
  normalizeProgressEvent({
    type: 'progress',
    event: 'retrieval_started',
    stage: 'retrieve',
    data: { attempt_id: 4 },
  }),
  {
    type: 'progress',
    stage: 'retrieval_started',
    message: '',
    data: { attempt_id: 4 },
  },
);

assert.deepEqual(
  normalizeProgressEvent({
    type: 'progress',
    event: 'answer_started',
    message: '正在生成答案……',
    data: {},
  }),
  {
    type: 'progress',
    stage: 'answer_started',
    message: '正在生成答案……',
    data: {},
  },
);

assert.equal(normalizeProgressEvent({ type: 'response', content: 'answer' }), null);
assert.equal(normalizeProgressEvent({ type: 'unmapped_internal_event' }), null);

applyProgressEvent(assistantMessage, {
  type: 'progress',
  stage: 'retrieval_started',
  message: '',
  data: {},
});
assert.equal(assistantMessage.progressStatus, '正在检索知识库……');
assert.equal(assistantMessage.progressDetail, '');
assert.equal(assistantMessage.isProgressVisible, true);

applyProgressEvent(assistantMessage, {
  type: 'progress',
  stage: 'retrieval_finished',
  message: '',
  data: { selected_documents: 5 },
});
assert.equal(assistantMessage.progressStatus, '正在整理检索结果……');
assert.equal(assistantMessage.progressDetail, '已找到 5 篇相关文档');

const backendRetrievalMessage = createAssistantProgressMessage();
applyProgressEvent(backendRetrievalMessage, normalizeProgressEvent({
  type: 'progress',
  stage: 'retrieval_finished',
  message: '已找到 5 篇相关文档',
  data: { selected_documents: 5 },
}));
assert.equal(backendRetrievalMessage.progressStatus, '正在整理检索结果……');
assert.equal(backendRetrievalMessage.progressDetail, '已找到 5 篇相关文档');

const positiveRetryStartedMessage = createAssistantProgressMessage();
applyProgressEvent(positiveRetryStartedMessage, {
  type: 'progress',
  stage: 'retrieval_finished',
  message: '',
  data: { selected_documents: 5 },
});
assert.equal(positiveRetryStartedMessage.progressDetail, '已找到 5 篇相关文档');
applyProgressEvent(positiveRetryStartedMessage, {
  type: 'progress',
  stage: 'retrieval_started',
  message: '',
  data: {},
});
assert.equal(positiveRetryStartedMessage.progressDetail, '');

const positiveRetryRewrittenMessage = createAssistantProgressMessage();
applyProgressEvent(positiveRetryRewrittenMessage, {
  type: 'progress',
  stage: 'retrieval_finished',
  message: '',
  data: { selected_documents: 5 },
});
assert.equal(positiveRetryRewrittenMessage.progressDetail, '已找到 5 篇相关文档');
applyProgressEvent(positiveRetryRewrittenMessage, {
  type: 'progress',
  stage: 'query_rewritten',
  message: '',
  data: {},
});
assert.equal(positiveRetryRewrittenMessage.progressDetail, '');

applyProgressEvent(assistantMessage, {
  type: 'progress',
  stage: 'answer_started',
  message: '正在生成答案……',
  data: {},
});
assert.equal(assistantMessage.progressStatus, '正在生成答案……');
assert.equal(assistantMessage.progressDetail, '已找到 5 篇相关文档');

hideAssistantProgress(assistantMessage);
assert.equal(assistantMessage.isProgressVisible, false);
assert.equal(assistantMessage.progressStatus, '');
assert.equal(assistantMessage.progressDetail, '');

const missingCountMessage = createAssistantProgressMessage();
applyProgressEvent(missingCountMessage, {
  type: 'progress',
  stage: 'retrieval_finished',
  message: '',
  data: {},
});
assert.equal(missingCountMessage.progressDetail, '检索完成，正在整理证据');

const retryMessage = createAssistantProgressMessage();
applyProgressEvent(retryMessage, {
  type: 'progress',
  stage: 'retrieval_finished',
  message: '',
  data: { selected_documents: 0 },
});
assert.equal(retryMessage.progressDetail, '已找到 0 篇相关文档');
applyProgressEvent(retryMessage, {
  type: 'progress',
  stage: 'query_rewritten',
  message: '',
  data: {},
});
assert.equal(retryMessage.progressDetail, '');
applyProgressEvent(retryMessage, {
  type: 'progress',
  stage: 'retrieval_finished',
  message: '',
  data: { selected_documents: 5 },
});
applyProgressEvent(retryMessage, {
  type: 'progress',
  stage: 'answer_started',
  message: '',
  data: {},
});
assert.equal(retryMessage.progressDetail, '已找到 5 篇相关文档');

console.log('chat progress verifier passed');
