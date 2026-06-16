const FALLBACK_STATUS = '正在分析问题……';

const STATUS_BY_STAGE = {
  route: '正在分析问题……',
  rag_plan_created: '正在规划检索策略……',
  strategy_selected: '正在规划检索策略……',
  retrieval_started: '正在检索知识库……',
  retrieval_finished: '正在整理检索结果……',
  evaluation_finished: '正在评估证据质量……',
  query_rewritten: '检索结果不足，正在调整查询……',
  topk_expanded: '检索结果不足，正在扩大检索范围……',
  metadata_filter_broadened: '检索结果不足，正在放宽筛选条件……',
  answer_started: '正在生成答案……',
};

const PROGRESS_STAGES = new Set(Object.keys(STATUS_BY_STAGE));
const RETRY_ADJUSTMENT_STAGES = new Set([
  'retrieval_started',
  'query_rewritten',
  'topk_expanded',
  'metadata_filter_broadened',
]);

export const createAssistantProgressMessage = () => ({
  role: 'assistant',
  content: '',
  progressStatus: FALLBACK_STATUS,
  progressDetail: '',
  isProgressVisible: true,
});

export const normalizeProgressEvent = (payload) => {
  if (!payload || typeof payload !== 'object') return null;

  if (payload.type === 'progress') {
    const stage = PROGRESS_STAGES.has(payload.stage) ? payload.stage : payload.event;
    if (!PROGRESS_STAGES.has(stage)) return null;
    return {
      type: 'progress',
      stage,
      message: payload.message || '',
      data: payload.data || {},
    };
  }

  const stage = payload.event || payload.type;
  if (!PROGRESS_STAGES.has(stage)) return null;

  return {
    type: 'progress',
    stage,
    message: payload.message || '',
    data: payload.data || {},
  };
};

export const applyProgressEvent = (message, progressEvent) => {
  if (!message || !progressEvent) return;

  const fallbackStatus = STATUS_BY_STAGE[progressEvent.stage];
  if (!fallbackStatus) return;

  message.progressStatus = fallbackStatus;
  message.isProgressVisible = true;

  if (progressEvent.stage === 'retrieval_finished') {
    const selectedDocuments = progressEvent.data?.selected_documents;
    const selectedDocumentCount = Number(selectedDocuments);
    message.progressDetail = progressEvent.message || (
      selectedDocuments !== undefined && selectedDocuments !== null && Number.isFinite(selectedDocumentCount)
        ? `已找到 ${selectedDocumentCount} 篇相关文档`
        : '检索完成，正在整理证据'
    );
    return;
  }

  message.progressStatus = progressEvent.message || fallbackStatus;

  if (progressEvent.stage === 'route') {
    message.progressDetail = '';
    return;
  }

  if (progressEvent.stage === 'rag_plan_created' || progressEvent.stage === 'strategy_selected') {
    message.progressDetail = '';
    return;
  }

  if (RETRY_ADJUSTMENT_STAGES.has(progressEvent.stage)) {
    message.progressDetail = '';
    return;
  }

};

export const hideAssistantProgress = (message) => {
  if (!message) return;

  message.isProgressVisible = false;
  message.progressStatus = '';
  message.progressDetail = '';
};
