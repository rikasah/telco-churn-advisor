# Telco Churn Advisor Revision Plan

Status: In progress (Milestones 0-2 initial slice)  
Prepared: 2026-08-26  
Working branch: `fix/security-runtime-validation`  
Scope: data, machine learning, RAG/agent, backend, frontend, security, observability, testing, and rollout

## 1. Executive Summary

Telco Churn Advisor currently demonstrates a complete ML application flow: ingest a public churn dataset, train and register a champion model in MLflow, score customers through FastAPI, explain predictions with SHAP, answer questions with an OpenRouter tool-calling agent and Chroma RAG, and present analytics in Streamlit.

The next revision should turn this from a prediction demo into an operational retention system. The product must not only identify high-risk customers. It must also prioritize work, recommend only valid retention actions, record what the team did, capture the outcome, and use those outcomes to improve future decisions.

The revision is divided into two tracks:

| Track | Purpose | Data source | Claim allowed |
|---|---|---|---|
| Demo track | Keep the project easy to run and evaluate | IBM static dataset plus clearly labeled synthetic events | Technical demonstration only |
| Production track | Produce measurable retention impact | Historical customer snapshots, events, actions, and outcomes | Business and model performance claims |

The recommended delivery order is:

1. Establish trustworthy baselines and secure service boundaries.
2. Correct data freshness, model lifecycle, and evaluation methodology.
3. Precompute versioned customer scores instead of rescoring the whole database per request.
4. Add a retention work queue, offer eligibility, and outcome tracking.
5. Improve multilingual RAG, tool reliability, and conversational continuity.
6. Add historical behavioral features, calibrated models, and business-driven thresholds.
7. Introduce next-best-action and uplift modeling only after intervention outcomes exist.

## 2. Product Vision

### 2.1 Product statement

Telco Churn Advisor helps retention teams decide which customers to contact, why they are at risk, what approved action to take, and whether the action successfully prevented churn.

### 2.2 Primary users

| User | Main job | Required experience |
|---|---|---|
| Retention agent | Contact and assist prioritized customers | Clear reason, approved offer, next step, case status |
| Retention manager | Allocate capacity and monitor outcomes | Queue health, segment performance, save rate, campaign ROI |
| Data scientist | Train and validate models | Reproducible datasets, evaluation gates, drift and fairness metrics |
| Administrator | Control access and configuration | Roles, offer catalog, audit logs, service health |

### 2.3 North-star metric

Use net retained value instead of raw prediction accuracy:

```text
net_retained_value = retained_revenue - offer_cost - contact_cost
```

Supporting product metrics:

| Metric | Meaning |
|---|---|
| Incremental churn reduction | Difference in churn between treated and comparable untreated customers |
| Save rate | Percentage of contacted at-risk customers who remain active after the outcome window |
| Offer acceptance rate | Percentage of eligible offers accepted |
| Cost per save | Retention campaign cost divided by successful saves |
| Queue completion rate | Percentage of assigned cases completed before due date |
| Time to action | Time from high-risk detection to first retention action |

## 3. Current-State Assessment

### 3.1 Existing strengths

- FastAPI, Streamlit, Postgres, MLflow, Chroma, and OpenRouter are already integrated.
- The training pipeline compares multiple classical models and tracks PR-AUC.
- The champion model is registered with an MLflow alias.
- SHAP provides per-customer explanation factors.
- The agent can call prediction, explanation, retrieval, risk summary, top-risk, and aggregation tools.
- The dashboard exposes model, traffic, risk, and segment analytics.
- Unit tests cover feature definitions, basic risk thresholds, ingestion cleaning, and agent tool routing.

### 3.2 Main limitations

| Area | Current limitation | Consequence |
|---|---|---|
| Data freshness | The same static IBM CSV is re-imported hourly | `ingested_at` appears fresh even when source data did not change |
| Prediction target | No snapshot date or prediction window | The model does not represent churn within a defined future period |
| Behavioral signals | Only static customer attributes are available | Important changes in usage, billing, service, and complaints are invisible |
| Evaluation | Candidate models share one fixed holdout | Model selection can overfit the evaluation split |
| Probability quality | No calibration evaluation | A displayed 70% score may not represent a real 70% likelihood |
| Thresholds | Risk bands are hardcoded at 0.25 and 0.50 | Queue size and business cost are not optimized |
| Model lifecycle | Champion model is cached until restart | A newly promoted model may not serve immediately |
| Batch analytics | All customers are rescored on several GET requests | Latency and compute grow with the dataset |
| RAG | Default embedding and only 10 document-level evaluation queries | Indonesian retrieval and answer faithfulness are weakly measured |
| Conversation | Backend receives only the current message | Follow-up questions lose prior conversational context |
| Retention workflow | No case, assignment, action, offer, or outcome records | Predictions cannot be converted into measurable work |
| Security | Internal endpoints and services lack production-grade access controls | Customer and operational data can be exposed |

## 4. Goals and Non-Goals

### 4.1 Goals

- Make data freshness and model version visible and trustworthy.
- Improve model selection, calibration, thresholding, and segment evaluation.
- Support reproducible daily batch scoring with immutable score history.
- Provide a retention work queue with assignment, approved actions, and outcomes.
- Prevent the LLM from inventing offers or directly making business decisions.
- Improve Indonesian RAG retrieval and evaluate complete answers, not only source files.
- Add role-based access, auditability, rate limits, and operational readiness checks.
- Preserve a simple Docker Compose demo path.

### 4.2 Non-goals for the first revision

- Do not add deep learning before tabular baselines are exhausted.
- Do not claim causal offer impact without randomized or defensible observational data.
- Do not automate outbound messages without human approval and consent controls.
- Do not split the codebase into many microservices unless scale proves it necessary.
- Do not use synthetic data to claim production model accuracy.

## 5. Engineering Principles

1. A prediction must include model version, score timestamp, and data snapshot version.
2. A recommendation must come from an approved offer catalog and eligibility rules.
3. The LLM may explain and orchestrate tools, but deterministic services own validation and decisions.
4. Offline model metrics and online business outcomes must be tracked separately.
5. Every retention action must be attributable to a user, case, offer, and timestamp.
6. Historical scores and outcomes are immutable audit records.
7. Expensive customer-wide scoring runs asynchronously or on schedule, not inside dashboard GET requests.
8. Protected attributes are evaluated for fairness and used in decisions only with explicit approval.
9. Demo-mode behavior must be visibly labeled as demo or synthetic.

## 6. Target Architecture

```text
                Customer data sources
       CRM | Billing | Usage | Tickets | Network
                        |
                        v
              Ingestion and validation
                        |
              dataset_versions/events
                        |
                        v
              Feature snapshot builder
                        |
               customer_snapshots
                        |
          +-------------+--------------+
          |                            |
          v                            v
  Scheduled training             Batch scoring job
          |                            |
   MLflow evaluation              customer_scores
          |                            |
   champion promotion                  |
          +-------------+--------------+
                        v
                 FastAPI backend
       auth | customers | queue | cases | offers
       explain | chat | analytics | monitoring
          |              |             |
          v              v             v
     Streamlit UI    RAG/LLM agent   Audit/metrics
          |
          v
 Retention action and outcome capture
          |
          v
 Training labels and uplift dataset
```

Deployment remains a modular monolith for the application layer. Postgres, MLflow, frontend, backend, and ingestion can stay as separate containers. Add a scheduled worker process for snapshot creation, scoring, and retraining rather than creating additional web services.

## 7. Target Data Model

Use migrations, not request-time `CREATE TABLE` or `ALTER TABLE` statements.

### 7.1 `dataset_versions`

Purpose: distinguish source freshness from ingestion execution time.

| Column | Type | Notes |
|---|---|---|
| `id` | UUID | Primary key |
| `source_name` | TEXT | CRM, IBM demo, billing, or other source |
| `source_version` | TEXT | Upstream version, date, or identifier |
| `content_hash` | TEXT | Detect unchanged data |
| `source_updated_at` | TIMESTAMPTZ | Actual source freshness |
| `ingested_at` | TIMESTAMPTZ | Time this system processed it |
| `row_count` | INTEGER | Validation and reconciliation |
| `status` | TEXT | pending, valid, rejected, loaded |
| `validation_report` | JSONB | Missing, invalid, duplicate, and drift results |

Indexes: unique `(source_name, content_hash)` and index on `source_updated_at`.

### 7.2 `customer_snapshots`

Purpose: support time-based prediction and reproducible features.

Key fields:

- `customer_id`
- `snapshot_at`
- `dataset_version_id`
- Existing categorical, boolean, and numeric features
- Usage aggregates for 7, 30, and 90 days
- Billing failures, late payments, and bill change percentages
- Ticket counts, complaint severity, and recent outage duration
- Contract and promotion expiry days
- Engagement and service-change counts
- `churn_within_30d`, populated only after the label window closes

Primary key: `(customer_id, snapshot_at)`.

### 7.3 `model_scores`

Purpose: serve fast, versioned, immutable scoring results.

| Column | Type | Notes |
|---|---|---|
| `id` | UUID | Primary key |
| `customer_id` | TEXT | Indexed |
| `snapshot_at` | TIMESTAMPTZ | Feature snapshot used |
| `scored_at` | TIMESTAMPTZ | Prediction time |
| `model_name` | TEXT | Registry model name |
| `model_version` | TEXT | Exact immutable version |
| `churn_probability` | DOUBLE | Calibrated probability |
| `risk_level` | TEXT | Derived from active threshold policy |
| `explanation` | JSONB | Top factors and values |
| `data_quality_flags` | JSONB | Missing, stale, or out-of-distribution indicators |

Indexes: `(customer_id, scored_at DESC)`, `(risk_level, churn_probability DESC)`, and `(model_version)`.

### 7.4 `retention_offers`

Purpose: prevent invented or unauthorized offers.

Key fields:

- `offer_id`, `name`, `description`
- `active_from`, `active_until`, `is_active`
- `cost_amount`, `discount_amount`
- `eligible_contracts`, `eligible_segments`
- `minimum_tenure`, `maximum_tenure`
- `requires_approval`
- `eligibility_rule_version`

### 7.5 `retention_cases`

Purpose: represent operational work.

Key fields:

- `case_id`, `customer_id`, `score_id`
- `priority`, `status`, `assigned_to`
- `recommended_offer_id`
- `created_at`, `due_at`, `closed_at`
- `reason_summary`
- `created_by_policy_version`

Allowed statuses: `open`, `assigned`, `contacted`, `offer_presented`, `resolved`, `closed_no_contact`.

### 7.6 `retention_actions`

Purpose: audit what happened in each case.

Key fields:

- `action_id`, `case_id`, `actor_id`
- `action_type`, `channel`, `offer_id`
- `action_at`, `notes`
- `customer_response`
- `approval_actor_id`, when required

### 7.7 `retention_outcomes`

Purpose: create labels for effectiveness and future uplift models.

Key fields:

- `case_id`, `customer_id`
- `measurement_at`
- `retained_30d`, `retained_60d`, `retained_90d`
- `offer_accepted`
- `revenue_retained`, `offer_cost`, `contact_cost`
- `outcome_source`

### 7.8 `audit_logs`

Store actor, action, resource type, resource ID, timestamp, request ID, and before/after metadata. Do not store LLM API keys, full credentials, or unrestricted message contents.

## 8. API Revision

Introduce `/v1` routes and migrate the frontend in the same release. Existing unversioned routes can be removed after frontend migration because the repository has no documented external consumer.

### 8.1 Prediction and customer APIs

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/v1/predictions` | On-demand score for one customer, with version metadata |
| `GET` | `/v1/customers/{customer_id}/risk` | Latest stored score and freshness |
| `GET` | `/v1/customers/{customer_id}/timeline` | Events, scores, actions, and outcomes |
| `GET` | `/v1/customers/{customer_id}/explanation` | Stored or generated explanation |
| `POST` | `/v1/scoring-jobs` | Admin-triggered batch scoring |
| `GET` | `/v1/scoring-jobs/{job_id}` | Batch status and counts |

Prediction response requirements:

```json
{
  "customer_id": "7590-VHVEG",
  "churn_probability": 0.7241,
  "risk_level": "high",
  "scored_at": "2026-08-26T10:00:00Z",
  "snapshot_at": "2026-08-25T00:00:00Z",
  "model": {"name": "telco_churn_model", "version": "12"},
  "threshold_policy_version": "2026-08-01",
  "data_quality_flags": []
}
```

### 8.2 Work queue and retention APIs

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/v1/work-queue` | Filtered and paginated prioritized cases |
| `POST` | `/v1/cases` | Create a case from an eligible score |
| `PATCH` | `/v1/cases/{case_id}` | Assign, transition status, or update due date |
| `POST` | `/v1/cases/{case_id}/actions` | Record customer contact or offer action |
| `POST` | `/v1/cases/{case_id}/outcomes` | Record measured result |
| `GET` | `/v1/offers/eligible` | Deterministic eligible offers for a customer |
| `GET` | `/v1/retention/summary` | Save rate, cost, retained value, and queue metrics |

Every list endpoint must support pagination and bounded page sizes.

### 8.3 Chat APIs

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/v1/chat/sessions` | Create a bounded conversation session |
| `POST` | `/v1/chat/sessions/{session_id}/messages` | Send a message with server-managed history |
| `DELETE` | `/v1/chat/sessions/{session_id}` | Clear session data |

The chat layer may call read-only prediction, explanation, documentation, eligible-offer, and aggregate tools. It must not create cases, send offers, or record outcomes without an explicit confirmed UI action.

### 8.4 Operational APIs

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health/live` | Process liveness only |
| `GET` | `/health/ready` | Database, MLflow model, and required storage readiness |
| `GET` | `/v1/system/status` | Authenticated operational summary |
| `GET` | `/v1/model/status` | Champion version, calibration, thresholds, and drift |

## 9. Data and Ingestion Plan

### 9.1 Phase A: trustworthy demo ingestion

Tasks:

- Compute SHA-256 before loading a source dataset.
- Skip loading when the content hash already exists.
- Record unchanged runs without changing `source_updated_at`.
- Persist validation reports and rejected row counts.
- Add schema checks for IDs, booleans, numerics, categories, nulls, duplicates, and finite values.
- Add a source mode flag: `demo_static`, `demo_synthetic`, or `production`.
- Label all synthetic UI data and metrics.
- Move schema creation to a migration tool such as Alembic.

Exit criteria:

- Re-ingesting the unchanged IBM CSV creates no new data version.
- Dashboard freshness reflects the source, not the scheduler execution time.
- Invalid batches are rejected atomically with a readable report.

### 9.2 Phase B: historical feature ingestion

Add connectors in this order:

1. Customer and contract snapshots.
2. Billing and payment events.
3. Support tickets and complaints.
4. Service usage aggregates.
5. Network incidents and outages.
6. Campaign, offer, and response history.

Use idempotent source event IDs and upserts. Keep raw source records or immutable staging tables long enough to reproduce feature snapshots.

### 9.3 Feature snapshot rules

- Generate snapshots daily or weekly at a fixed timestamp.
- Use only data known at snapshot time.
- Define churn as an event occurring within 30 days after snapshot time.
- Do not train on snapshots whose label window has not closed.
- Exclude post-churn or post-intervention leakage.
- Version feature definitions and store the version with every score.

## 10. Model Accuracy Plan

### 10.1 Baseline protocol

Before changing algorithms, save a reproducible baseline containing:

- Dataset version and feature version
- Train, validation, and test date ranges
- Class balance
- PR-AUC, ROC-AUC, precision, recall, F1
- Brier score and log loss
- Calibration plot
- Confusion matrix at active thresholds
- Precision and recall at top 1%, 5%, 10%, and daily queue capacity
- Metrics by contract, gender, senior status, internet service, and tenure band
- Training duration, artifact size, and serving latency

### 10.2 Split strategy

| Data availability | Required split |
|---|---|
| Static IBM demo | Stratified train/validation/test plus repeated cross-validation |
| Historical production | Time-based train, validation, and final test windows |

The final test set must not select models or thresholds.

### 10.3 Candidate models

Evaluate in this order:

1. Logistic Regression baseline.
2. Existing Random Forest and Gradient Boosting baselines.
3. CatBoost for categorical tabular data.
4. LightGBM or XGBoost if operationally acceptable.

Do not promote a more complex model unless it improves the agreed operational metric and remains explainable enough for retention decisions.

### 10.4 Probability calibration

Evaluate uncalibrated, Platt-scaled, and isotonic-calibrated probabilities on validation data. Select calibration using Brier score and calibration error, then report final calibration once on the test set.

### 10.5 Threshold policy

Replace hardcoded risk bands with a versioned threshold policy.

Inputs:

- Daily contact capacity
- Customer lifetime value
- Estimated contact cost
- Offer cost
- Minimum acceptable precision
- Segment constraints

The initial implementation may choose thresholds from validation data to maximize expected value. Store the chosen threshold policy in MLflow and the database.

### 10.6 Champion promotion gate

A challenger may become champion only if all required checks pass:

- Primary operational metric improves by the configured minimum.
- PR-AUC does not regress beyond tolerance.
- Calibration does not regress beyond tolerance.
- No critical segment regression occurs.
- Feature schema is compatible with serving.
- Latency and artifact size stay within limits.
- Integration smoke tests pass against the registered artifact.

### 10.7 Model serving lifecycle

- Cache the model by immutable MLflow version, not only by alias name.
- Poll the champion alias or receive a promotion signal.
- Load and smoke-test the new version before swapping it into service.
- Keep the prior model in memory for immediate rollback.
- Invalidate SHAP explainers when the active model version changes.
- Expose active model version through readiness and prediction responses.

### 10.8 Drift monitoring

Monitor:

- Feature missingness and schema violations
- Population Stability Index or equivalent distribution shifts
- Prediction distribution and risk-band distribution
- Calibration and performance when labels mature
- Segment-level drift
- Data staleness

Alert thresholds must be configured and versioned rather than hardcoded.

## 11. Scoring and Explanation Plan

### 11.1 Batch scoring

- Schedule daily scoring after the latest valid snapshot is available.
- Score each customer once per model and snapshot version.
- Write results to `model_scores` in batches.
- Record failures separately and make the job resumable.
- Serve top-risk, risk summary, and segment analytics from stored scores.
- Keep on-demand prediction only for debugging or newly changed customers.

### 11.2 Explanation generation

- Generate top factors during batch scoring or lazily cache them by score ID.
- Use categories from the training schema, not a random explanation background sample.
- Store raw feature value, contribution direction, magnitude, and model version.
- Add an uncertainty or data-quality warning when the customer is out of distribution.
- Present SHAP as model reasoning, not proof of causality.

### 11.3 What-if simulation

Add only after the core workflow is stable.

Rules:

- Restrict editable features to actionable fields.
- Display original and simulated scores with the same model version.
- Label output as a scenario estimate, not guaranteed churn reduction.
- Do not automatically translate score changes into offer effectiveness.

## 12. RAG and Agent Plan

### 12.1 Retrieval improvements

- Replace the uncontrolled default embedding with a pinned multilingual embedding model.
- Evaluate multilingual E5, BGE-M3, or a selected provider embedding.
- Add BM25 keyword retrieval for exact product, contract, and offer terms.
- Fuse vector and keyword results.
- Add reranking when baseline retrieval is insufficient.
- Store document version, effective date, section title, and access classification in metadata.
- Reindex automatically when document hashes change.
- Exclude expired policy versions from normal retrieval.

### 12.2 Knowledge governance

- Require review before policy documents are indexed.
- Keep policy effective dates and owners.
- Reject or quarantine documents containing suspicious instruction-like content.
- Separate public FAQs from internal policy documents.
- Enforce document access filters before retrieval.

### 12.3 Agent reliability

- Define Pydantic schemas for every tool input and output.
- Bound tool arguments, result sizes, and loop iterations.
- Return structured tool errors to the model without leaking stack traces.
- Add request IDs and tool-call audit events.
- Apply timeouts and retry policy to OpenRouter and internal tools.
- Use deterministic offer eligibility tools instead of generating offers from free text.
- Redact or minimize customer identifiers before sending data to an external LLM where possible.

### 12.4 Conversation memory

- Store only a bounded number of recent turns.
- Summarize older context when needed.
- Enforce a total token limit.
- Tie sessions to authenticated users.
- Provide explicit clear-session controls.
- Define retention and deletion policy for message data.

### 12.5 RAG and agent evaluation

Expand the evaluation set from 10 queries to at least 100 cases covering:

- Indonesian paraphrases and informal language
- Typographical errors
- Exact policy terms
- Multi-document questions
- Questions with no valid answer
- Conflicting or expired policies
- Direct and indirect prompt injection
- Correct and incorrect customer context
- Tool selection and argument correctness
- Citation precision and answer faithfulness

Required metrics:

- Recall@k, MRR, and nDCG for retrieval
- Citation precision and citation coverage
- Grounded answer pass rate
- Unsupported-claim rate
- Correct tool-selection rate
- Tool argument validation failure rate
- Prompt-injection resistance pass rate

## 13. Retention Workflow Plan

### 13.1 Work queue creation

Create cases from the latest model scores using a versioned prioritization policy:

```text
priority_score = churn_probability
               * customer_value_weight
               * contactability_weight
               * data_quality_weight
```

Do not include customers who are already churned, opted out, recently contacted, or otherwise ineligible.

### 13.2 Queue capabilities

- Filter by risk, value, segment, due date, assignee, and case status.
- Sort by expected priority rather than probability alone.
- Support pagination and CSV export.
- Show model version, score age, and data-quality warnings.
- Prevent duplicate open cases for the same customer and policy window.
- Record assignment and state transitions in the audit log.

### 13.3 Offer eligibility

The eligibility engine receives customer data and returns approved offers with reasons. It must be deterministic and separately tested. The LLM can explain returned offers but cannot add new ones.

### 13.4 Outcome measurement

- Define 30, 60, and 90-day outcome windows.
- Schedule outcome resolution after windows close.
- Distinguish contacted, untreated, accepted, rejected, and unreachable customers.
- Capture costs and retained revenue.
- Preserve untreated comparison groups for impact measurement.

## 14. Frontend Revision

Continue using Streamlit for the next revision unless user volume or workflow complexity proves it inadequate.

### 14.1 Information architecture

| Page | Purpose |
|---|---|
| Overview | Data freshness, model status, queue size, save rate, retained value |
| Work Queue | Prioritized and assignable retention cases |
| Customer Detail | Timeline, risk, explanation, eligible offers, actions, outcomes |
| Campaigns | Segment filters, campaign cohorts, and performance |
| Model Monitor | Metrics, calibration, drift, fairness, version history |
| Knowledge | Policy documents, versions, retrieval evaluation |
| Chat Assistant | Contextual explanation and policy assistance |
| Admin | Users, roles, offers, thresholds, and system settings |

### 14.2 UX requirements

- Clearly distinguish actual churn, predicted risk, simulated score, and intervention outcome.
- Display score age and model version near every risk value.
- Require confirmation before recording an action or presenting an offer.
- Show why an offer is eligible or ineligible.
- Preserve filters and pagination in queue navigation.
- Provide empty, loading, stale-data, partial-failure, and permission-denied states.
- Maintain mobile-readable customer detail while optimizing queue work for desktop.
- Keep charts secondary to actionable tables and cases.

## 15. Security and Privacy Plan

### 15.1 Authentication and authorization

Roles:

| Role | Access |
|---|---|
| Agent | Assigned queue, customer detail, approved actions |
| Manager | Team queue, campaigns, aggregate outcomes |
| Data scientist | Model and data monitoring without unnecessary customer PII |
| Admin | Users, roles, offers, configuration, audit |

Enforce authorization in the backend, not only in Streamlit.

### 15.2 Network boundaries

- Publish only the frontend or a controlled reverse proxy.
- Keep Postgres, MLflow, and ingestion internal to the Docker network.
- Remove default production credentials.
- Store secrets outside committed environment files.
- Add TLS at the public boundary.

### 15.3 API protection

- Add request size limits.
- Add per-user and per-route rate limits.
- Validate all query, body, and tool parameters.
- Add pagination caps.
- Prevent unrestricted analytics exports.
- Return generic external errors and log internal details securely.

### 15.4 Data and LLM privacy

- Define which customer fields may leave the system.
- Minimize identifiers and attributes sent to OpenRouter.
- Document provider retention and training policies.
- Add message and audit retention periods.
- Provide deletion procedures for conversation data.
- Avoid storing complete prompts when metadata is sufficient for monitoring.

## 16. Observability and Operations

### 16.1 Metrics

Application metrics:

- Request count, error rate, and p50/p95/p99 latency
- Tool calls, failures, retries, and timeout rates
- Queue depth and case transition counts
- OpenRouter latency and token usage
- Scoring job duration, throughput, and failure count

ML metrics:

- Active model and feature versions
- Score distribution
- Calibration and delayed-label performance
- Drift and segment metrics

Data metrics:

- Source freshness
- Last changed content hash
- Valid, rejected, duplicate, and missing rows
- Snapshot generation lag

### 16.2 Logging

- Use structured JSON logs.
- Include request ID, user ID, route, model version, and duration.
- Do not log credentials or unrestricted customer payloads.
- Move request-log schema setup to migrations.
- Add indexes and retention policies.

### 16.3 Health checks

- Liveness checks only process health.
- Readiness verifies Postgres, active model availability, and required storage.
- Optional dependencies such as OpenRouter should report degraded state without hiding core prediction readiness.

## 17. Testing Strategy

### 17.1 Unit tests

- Ingestion schema and content validation
- Feature calculations and leakage guards
- Threshold and priority policy
- Offer eligibility
- Case state transitions
- Model version reload
- Tool schemas and argument bounds
- Retrieval chunking, indexing, and access filters

### 17.2 Integration tests

- Ingestion to Postgres with transactional rollback
- Training to MLflow registration
- Champion promotion and serving reload
- Batch scoring to `model_scores`
- FastAPI routes with Postgres and mocked OpenRouter
- Case, action, and outcome lifecycle
- Document update to retrieval result

### 17.3 End-to-end tests

1. Ingest a versioned demo dataset.
2. Train and promote a model.
3. Score customers.
4. Create a retention case.
5. Assign and record an action.
6. Record an outcome.
7. Verify analytics update.
8. Ask chat to explain the same customer using approved sources and tools.

### 17.4 ML tests

- Dataset split reproducibility
- No future-data leakage
- Feature schema compatibility
- Baseline comparison gate
- Calibration gate
- Segment regression gate
- Prediction invariants and finite outputs

### 17.5 Security tests

- Anonymous access denied to protected endpoints
- Role matrix enforcement
- Rate and payload limits
- Prompt injection suite
- Document access-control filtering
- Secret scanning and dependency scanning
- Unauthorized offer or action mutation rejected

### 17.6 CI pipeline

Required CI jobs:

1. Ruff lint and formatting check.
2. Type checking for backend and ingestion.
3. Unit tests with coverage.
4. Integration tests with Postgres.
5. API schema and migration checks.
6. Dependency and secret scans.
7. Docker image build.
8. Small deterministic ML smoke test.
9. RAG regression evaluation with no network dependency.

## 18. File and Module Revision Map

Keep modules cohesive and avoid unnecessary service splitting.

| Path | Planned revision |
|---|---|
| `ingestion/pipeline.py` | Source versioning, atomic validation, staging, historical snapshots |
| `ingestion/main.py` | Single-worker scheduling, job IDs, structured status, readiness |
| `backend/train.py` | Split protocol, calibration, candidate evaluation, promotion gates |
| `backend/model.py` | Version-aware loading, metadata responses, safe hot reload |
| `backend/explain.py` | Version-aware explainer cache and training-schema categories |
| `backend/system_status.py` | Query stored scores instead of rescoring all customers |
| `backend/rag.py` | Multilingual embeddings, hybrid retrieval, metadata and versions |
| `backend/eval_rag.py` | Larger eval set and answer/tool/citation metrics |
| `backend/agent.py` | Typed tools, bounded history, eligible-offer tool, audit events |
| `backend/app.py` | Versioned routers, auth dependencies, pagination, readiness |
| `backend/logging_store.py` | Remove request-time DDL, structured events, retention support |
| `frontend/app.py` | Work queue, customer timeline, outcomes, model monitor, role-aware UI |
| `backend/db/` | New SQLAlchemy models, repositories, and migrations |
| `backend/services/` | Scoring, retention policy, offer eligibility, case workflow |
| `backend/routers/` | Versioned API routers grouped by domain |
| `worker/` | Snapshot, scoring, outcome-resolution, and retraining jobs |
| `tests/` | Unit, integration, ML, RAG, security, and workflow coverage |

Only introduce the new directories when the related phase begins. Do not perform a structure-only refactor before behavior is implemented.

## 19. Delivery Milestones

### Milestone 0: Baseline and planning lock

Effort: Small

Deliverables:

- Baseline model and RAG reports committed as reproducible scripts or fixtures.
- Current API and data contracts documented.
- Product metrics and outreach capacity assumptions recorded.
- Demo and production data modes explicitly separated.

Exit criteria:

- The team can rerun the same baseline and get equivalent results.
- No later phase starts without a measurable comparison point.

### Milestone 1: Secure and trustworthy runtime

Effort: Medium

Deliverables:

- Authentication and role enforcement.
- Internal-only Postgres, MLflow, and ingestion ports.
- Liveness/readiness separation.
- Input, pagination, and rate limits.
- Dataset content hashes and true freshness.
- Alembic migration baseline.

Exit criteria:

- Anonymous users cannot access customer or operational endpoints.
- Unchanged datasets do not appear newly updated.
- Readiness fails when the database or active model is unavailable.

### Milestone 2: Reproducible model lifecycle

Effort: Medium

Deliverables:

- Train/validation/test protocol.
- Calibration and capacity-aware thresholds.
- Champion promotion gates.
- Version-aware model and SHAP reload.
- Model metadata in predictions.

Exit criteria:

- A promoted champion is served without restart and can roll back.
- Every prediction is attributable to model, feature, and dataset versions.

### Milestone 3: Batch scores and fast analytics

Effort: Medium

Deliverables:

- `customer_snapshots` and `model_scores` migrations.
- Scheduled, resumable batch scoring.
- Top-risk, summary, and aggregate endpoints read stored scores.
- Scoring job status and monitoring.

Exit criteria:

- Dashboard requests no longer score the full database.
- Latest and historical scores are queryable and versioned.

### Milestone 4: Retention operations MVP

Effort: Large

Deliverables:

- Offer catalog and eligibility service.
- Work queue, case assignment, action logging, and outcomes.
- Customer detail timeline.
- Manager outcome dashboard.
- Audit logs for case mutations.

Exit criteria:

- A user can complete the full case lifecycle without editing the database.
- The system can calculate save rate, cost per save, and retained value.

### Milestone 5: RAG and agent quality

Effort: Medium

Deliverables:

- Pinned multilingual embeddings and hybrid retrieval.
- Versioned policy metadata and access controls.
- Typed tools and bounded conversation sessions.
- At least 100 evaluation cases.
- Citation, grounding, tool-selection, and injection metrics.

Exit criteria:

- RAG and agent quality gates run in CI.
- The agent only recommends offers returned by eligibility tools.

### Milestone 6: Historical behavioral model

Effort: Large and data-dependent

Deliverables:

- Historical snapshots and 30-day labels.
- Usage, billing, complaint, outage, and engagement trends.
- Time-based model evaluation.
- Drift and delayed-label monitoring.

Exit criteria:

- Production evaluation uses future outcomes relative to snapshot time.
- No synthetic data contributes to production claims.

### Milestone 7: Next-best-action experimentation

Effort: Large and outcome-data-dependent

Deliverables:

- Randomized or defensible treatment/control cohorts.
- Offer-level incremental impact reporting.
- Uplift or treatment-effect model when sample size permits.
- Expected-value prioritization using CLV and action cost.

Exit criteria:

- Recommendations optimize incremental retained value, not churn risk alone.
- Product reporting distinguishes correlation from causal lift.

## 20. Success Gates

Targets must be confirmed after Milestone 0 establishes baselines.

Proposed initial gates:

| Area | Gate |
|---|---|
| Model ranking | PR-AUC and top-capacity precision do not regress from baseline |
| Calibration | Brier score and calibration error improve or remain within tolerance |
| Fairness | No critical segment regression without documented approval |
| Serving | Prediction p95 meets the agreed target and includes version metadata |
| Analytics | Work queue and summary queries use stored scores and bounded pagination |
| RAG retrieval | Recall@3 at least 0.90 on the approved evaluation set |
| RAG answers | Citation precision at least 0.95 and unsupported-claim rate below agreed tolerance |
| Agent tools | Correct tool selection and valid arguments at least 0.95 on evaluation cases |
| Security | Protected routes reject anonymous and unauthorized requests |
| Product | Full case and outcome lifecycle works with complete audit history |

Business success gates require real outcome data and should include incremental churn reduction and net retained value.

## 21. Rollout and Rollback

### 21.1 Rollout sequence

1. Deploy database migrations with no traffic behavior changes.
2. Backfill dataset versions and initial score history.
3. Run old and new scoring paths in shadow mode.
4. Compare scores, latency, calibration, and segment metrics.
5. Switch read APIs to stored scores behind a feature flag.
6. Enable work queue for an internal pilot group.
7. Enable outcome tracking before expanding campaigns.
8. Enable new RAG and agent behavior after evaluation gates pass.
9. Retire old routes and rescoring paths after the frontend migration is verified.

### 21.2 Rollback requirements

- Keep the prior MLflow champion version available.
- Keep prior threshold and prioritization policies versioned.
- Make schema migrations backward-safe within the rollout window.
- Feature-flag new scoring reads, work queue creation, and agent tools.
- Never delete historical scores, actions, or outcomes during rollback.

## 22. Main Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Static data is mistaken for production evidence | Label demo mode and prohibit production claims |
| Historical data contains label leakage | Enforce snapshot-time feature tests and reviews |
| High-risk targeting wastes retention budget | Optimize queue by value and later by uplift |
| LLM invents policy or offers | Deterministic offer catalog, typed tools, grounded citations |
| Model promotion breaks serving | Artifact smoke test, version-aware hot swap, rollback model |
| Protected groups receive uneven treatment | Segment metrics, policy review, audit reporting |
| Queue overwhelms agents | Capacity-aware thresholds and daily case caps |
| Outcome data is biased | Preserve untreated cohorts and randomize when possible |
| Streamlit becomes limiting | Reassess only after operational workflow usage is measured |

## 23. Definition of Done

The revision is complete when:

- Data freshness reflects source changes rather than scheduler runs.
- Every prediction records dataset, feature, model, and threshold versions.
- Model selection uses a reproducible protocol with calibration and segment checks.
- Champion promotion and rollback work without restarting the backend.
- Customer-wide analytics read precomputed scores.
- Authenticated users can manage retention cases, approved offers, actions, and outcomes.
- Business dashboards report save rate, cost per save, and retained value.
- The RAG and agent evaluation suite covers retrieval, grounding, citations, tools, and injection.
- The LLM cannot create an unauthorized offer or mutate a case without confirmation.
- CI covers lint, types, unit tests, integration tests, migrations, security, Docker build, ML smoke tests, and RAG regressions.
- Deployment and rollback procedures have been tested.
- Demo and production metrics are clearly separated.

## 24. Recommended First Implementation Slice

Implement the following vertical slice before broader feature work:

1. Add `dataset_versions`, `model_scores`, `retention_offers`, `retention_cases`, `retention_actions`, and `retention_outcomes` migrations.
2. Make ingestion hash-aware so unchanged data does not appear fresh.
3. Add version-aware champion loading and prediction metadata.
4. Add one scheduled batch-scoring job and switch top-risk endpoints to stored scores.
5. Add a minimal authenticated work queue.
6. Add one deterministic offer eligibility rule.
7. Complete one case through assignment, action, and 30-day outcome entry.
8. Add integration tests for the complete slice.

This slice proves the core product loop without waiting for advanced models or production data connectors. It creates the outcome data needed for every later accuracy and next-best-action improvement.
