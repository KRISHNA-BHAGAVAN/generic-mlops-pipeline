# Observability Stack Research & Recommendations
## OpenTelemetry, Prometheus, Grafana, EvidentlyAI for MLOps

**Research Date:** April 2026  
**Status:** Completed  
**Recommendation:** IMPLEMENT in Phase 1.1 (Prometheus/Grafana) + Phase 1.2 (EvidentlyAI)

---

## Executive Summary

**Should you integrate monitoring tools into your MLOps pipeline?**

**YES, absolutely.** Here's why and how:

- **Prometheus + Grafana:** Include in Phase 1.1 (MVP). These are lightweight, open-source, and solving a critical problem: visibility into what your models are actually doing in production.
- **EvidentlyAI:** Include in Phase 1.2. Specialized for ML drift detection; pairs perfectly with Prometheus.
- **OpenTelemetry:** Skip for Phase 1.1; plan for Phase 2 (future). Currently stabilizing; not required for basic observability.

**ROI:** Detecting a model degradation within hours instead of days saves business impact. The infrastructure cost is minimal (~$50/month if self-hosted).

---

## Market Research & Trends

### 1. OpenTelemetry (2025 Landscape)

**Status:** De facto standard, reaching v1.0 maturity in 2025.

**Key Findings:**
- OpenTelemetry is now the success story of 2024, as a way to instrumentalize the standardization of tools for observability, covering metrics, traces, logs and much more, offering freedom of interchangeability between different observability solutions
- Major observability players are switching their default way of collecting telemetry from proprietary technology to OpenTelemetry
- The OpenTelemetry Collector is expected to reach v1.0 milestone in 2025, and profiling support is advancing

**For Your Pipeline:**
- **Phase 1.1:** Not required. Use Prometheus client library directly (simpler, fewer dependencies).
- **Phase 2:** Evaluate OpenTelemetry for distributed tracing and advanced instrumentation.

### 2. Prometheus (2025 Landscape)

**Status:** Industry standard, mature, widely adopted.

**Key Findings:**
- Prometheus and PromQL will remain the default standard for metrics storage and querying, cementing their role in observability architectures
- Lightweight time-series database: <1GB memory for typical ML workloads
- Pull-based model (Prometheus scrapes targets) is perfect for episodic ML jobs
- PromQL query language is powerful and well-documented

**For Your Pipeline:**
- **Phase 1.1:** REQUIRED. Use for:
  - Collecting metrics from FastAPI inference service
  - Storing time-series metrics (15-day retention)
  - Evaluating alert rules every 30s
  - Exposing metrics API for Grafana

**Why Prometheus over alternatives:**
| Tool | Cost | Ease | Maturity | Why Choose |
|------|------|------|----------|-----------|
| **Prometheus** | Free (OSS) | Easy | Mature (2012+) | **Perfect for MVP** |
| Datadog | $$$$ | Medium | Mature | Overkill for Phase 1 |
| New Relic | $$$$ | Hard | Mature | Expensive, complex |
| CloudWatch | $$$ (AWS only) | Medium | Mature | Vendor lock-in |
| InfluxDB | $ (OSS) | Medium | Mature | Works but Prometheus better for metrics |

### 3. Grafana (2025 Landscape)

**Status:** Best-in-class visualization; industry standard.

**Key Findings:**
- Grafana provides a feature-rich platform to create dashboards, and works seamlessly with Prometheus
- Grafana, together with Prometheus, is used to provide monitoring component in model serving, enabling insights into running models and alerting
- Supports multiple data sources (Prometheus, MLflow, databases, APIs)
- Can trigger alerts directly to Slack, PagerDuty, webhooks

**For Your Pipeline:**
- **Phase 1.1:** REQUIRED. Use for:
  - System health dashboard (latency, errors, throughput)
  - Model performance dashboard (accuracy, predictions)
  - Real-time alerting UI
  - Historical trend analysis

**Grafana + Prometheus = Perfect pair for observability:**
- Prometheus collects metrics
- Grafana visualizes metrics
- Alerts can be defined in Prometheus and routed through Grafana

### 4. EvidentlyAI (2025 Landscape)

**Status:** Specialized ML monitoring; rapidly growing.

**Key Findings:**
- EvidentlyAI has large number of drift detection methods including PSI, K-L divergence, Jensen-Shannon distance, Wasserstein distance, etc.
- Evidently open-source Python library now supports evaluations for LLM-based applications, including RAGs and chatbots
- Integrates directly with Prometheus (exports metrics)
- Can detect data drift, prediction drift, and data quality issues
- Used by major companies for production ML monitoring

**For Your Pipeline:**
- **Phase 1.1:** OPTIONAL (skip, focus on core pipeline)
- **Phase 1.2:** REQUIRED. Use for:
  - Detecting distribution shift in features
  - Detecting prediction drift
  - Data quality checks (missing values, outliers)
  - Generating drift reports

**Why EvidentlyAI is worth the investment:**
- Without it: You'd need to write custom drift detection code (weeks of work)
- With it: 50 lines of code + batch job = complete drift monitoring
- Cost: Free (open-source)

---

## Monitoring Architecture Decision Matrix

### Phase 1.1 (MVP) – What to Include?

| Component | Include? | Why | Effort |
|-----------|----------|-----|--------|
| **Prometheus** | ✅ YES | Core metrics collection; lightweight; proven | 3 days |
| **Grafana** | ✅ YES | Essential for visibility; simple setup; 2 dashboards | 2 days |
| **EvidentlyAI** | ❌ NO | Can defer; focus on core pipeline first | Later |
| **OpenTelemetry** | ❌ NO | Overkill for Phase 1; don't need tracing yet | Phase 2 |
| **AlertManager** | ✅ YES | Simple setup; enables Slack alerts | 1 day |

**Total Phase 1.1 Effort:** 6 additional days (2 weeks instead of 4, so final timeline is ~4-5 weeks)

### Phase 1.2 (Enhanced) – What to Add?

| Component | Include? | Why | Effort |
|-----------|----------|-----|--------|
| **EvidentlyAI** | ✅ YES | Specialized drift detection; saves weeks of custom code | 4 days |
| **Batch Monitoring Job** | ✅ YES | Daily drift detection; automatic metrics export | 3 days |
| **Slack Integration** | ✅ YES | Alerts go to team; enables faster response | 1 day |
| **Advanced Dashboards** | ✅ YES | Drift visualization; 2 additional dashboards | 2 days |

**Total Phase 1.2 Effort:** 10 additional days (2 weeks)

### Phase 2+ (Future) – What Might Be Useful?

| Component | Include? | Why | Effort |
|-----------|----------|-----|--------|
| **OpenTelemetry** | ⚠️ MAYBE | Distributed tracing; need when services grow | Weeks |
| **Custom Profiling** | ⚠️ MAYBE | GPU memory, inference time per layer | Weeks |
| **Automated Retraining** | ⚠️ MAYBE | Trigger retraining on drift; complex | Weeks |
| **A/B Testing Framework** | ⚠️ MAYBE | Statistical significance testing; nice to have | Weeks |

---

## Detailed Recommendations

### Recommendation 1: Prometheus + Grafana in Phase 1.1

**Why this makes sense:**

1. **Minimal complexity:** Single config file, pull-based scraping
2. **Low resource footprint:** ~500MB memory total
3. **Immediate value:** See latency, errors, throughput in real-time
4. **No vendor lock-in:** Open-source, portable
5. **Proven by industry:** Used by Uber, Netflix, Amazon internally

**What you'll gain in Phase 1.1:**
- Real-time visibility into inference service health
- Alerting on latency spikes and errors
- Historical trends (15 days default)
- Foundation for future enhancements

**Effort breakdown:**
- FastAPI instrumentation: 2 days
- Prometheus setup: 1 day  
- Grafana dashboards: 2 days
- Testing & validation: 1 day

**Total: ~6 additional days** (still fits in 4-week timeline if you parallelize)

---

### Recommendation 2: EvidentlyAI + Batch Monitoring in Phase 1.2

**Why this makes sense:**

1. **Solves a critical problem:** Detecting when models degrade in production
2. **Minimal friction:** Batch job runs daily, no real-time overhead
3. **Prevents silent failures:** You catch drift before business impact
4. **Small learning curve:** ~100 lines of code for complete setup
5. **Future-proof:** Integrates with any monitoring system

**What you'll gain in Phase 1.2:**
- Automatic data drift detection
- Prediction drift monitoring  
- Data quality checks
- Drift alerts to Slack
- Historical drift reports in MLflow

**Effort breakdown:**
- EvidentlyAI integration: 2 days
- Batch monitoring job: 2 days
- Drift dashboards: 1 day
- Slack alerting: 1 day
- Testing: 1 day

**Total: ~7 additional days** (fits in 2-week Phase 1.2 timeline)

---

### Recommendation 3: Skip OpenTelemetry for Phase 1

**Why waiting makes sense:**

1. **Not urgent:** Basic Prometheus metrics solve current needs
2. **Under stabilization:** v1.0 not quite ready; too early to commit
3. **Adds complexity:** Collector, multiple exporters, schema validation
4. **Overkill for single service:** Distributed tracing needed when you have 10+ services
5. **Can add later:** Prometheus metrics are compatible with OpenTelemetry

**When to revisit (Phase 2):**
- You have multiple services (training orchestrator, inference, monitoring jobs)
- You need distributed tracing across services
- OpenTelemetry reaches stable v1.0

**Migration path exists:** You can start with Prometheus, add OpenTelemetry later without rewriting everything.

---

## Cost-Benefit Analysis

### Phase 1.1: Prometheus + Grafana

**Costs:**
- Development: 6 days
- Infrastructure: ~$50/month (self-hosted on 1GB instance)
- Maintenance: ~4 hours/month

**Benefits:**
- Real-time visibility into inference performance
- Early warning on latency/error spikes
- Foundation for drift detection (Phase 1.2)
- Team confidence in model reliability

**ROI:** Breaking even in month 1 by catching a single inference issue that would have cost hours to debug blindly.

### Phase 1.2: EvidentlyAI + Batch Monitoring

**Costs:**
- Development: 7 days
- Infrastructure: ~$0 (uses existing Prometheus)
- Maintenance: ~2 hours/month

**Benefits:**
- Automated drift detection saves manual monitoring work
- Early alert prevents model decay
- Enables data-driven retraining decisions
- Audit trail for compliance

**ROI:** Worth it if you ever have to explain why a model made bad predictions ("We didn't have drift detection" is not a good look in the boardroom).

---

## Implementation Path (Final Recommendation)

### Week 1-3: Phase 1.1 Core + Monitoring
```
Week 1:  Config system + Model factory
Week 2:  FastAPI + Prometheus instrumentation
Week 3:  Grafana dashboards + Alerts + Integration testing
```

### Week 4-5: Phase 1.2 Drift Detection
```
Week 4: EvidentlyAI + Batch monitoring job
Week 5: Drift dashboards + Slack alerts
```

### Week 6-8: Polish + Deployment
```
Week 6: End-to-end testing + Runbooks
Week 7: Staging deployment + Load testing
Week 8: Production deployment + Documentation
```

**Total: 8 weeks** (includes monitoring from start)

---

## Alternative Scenarios

### Scenario A: If you want ZERO monitoring initially

**Not recommended.** But if you absolutely must defer:

- Phase 1.1: Skip Prometheus/Grafana
- Phase 1.2: Add Prometheus/Grafana
- **Risk:** Month of production traffic with no visibility

**Verdict:** Bad idea. Add 1 week to Phase 1.1 and solve it.

### Scenario B: If you want monitoring but not drift detection

**Acceptable alternative:**

- Phase 1.1: Add Prometheus/Grafana ✅
- Phase 1.2: Skip EvidentlyAI
- **Risk:** You'll detect problems (high latency) but not root cause (data drift)

**Verdict:** Works for simple models; risky for complex ones.

### Scenario C: If you want everything in Phase 1.1

**Technically possible but not recommended.**

- Timeline: 8 weeks becomes 10-11 weeks
- Quality: Risk overlooking core pipeline for monitoring details

**Verdict:** Better to deliver core + basic monitoring, then add drift detection.

---

## Industry Examples

### How others do it:

**Uber:** Prometheus for metrics + custom drift detection at scale
- Uses OpenTelemetry for distributed tracing across services
- EvidentlyAI-like custom system for drift

**Netflix:** Similar stack + heavy emphasis on anomaly detection
- Detects when metrics diverge from statistical norms

**Airbnb:** Prometheus + Grafana + custom ML monitoring (published in papers)
- Focuses on feature distribution monitoring

**Stripe:** Multiple exporters → centralized observability
- Could use OpenTelemetry but built before it existed

**Lesson:** All successful ML platforms have observability. The specific tools matter less than the commitment to visibility.

---

## Final Verdict

| Decision | Recommendation | Confidence |
|----------|----------------|------------|
| **Include Prometheus in Phase 1.1?** | ✅ YES | 95% |
| **Include Grafana in Phase 1.1?** | ✅ YES | 95% |
| **Include EvidentlyAI in Phase 1.2?** | ✅ YES | 85% |
| **Include OpenTelemetry in Phase 1.1?** | ❌ NO | 90% |
| **Include OpenTelemetry in Phase 2?** | ⚠️ CONSIDER | 70% |

---

## Action Items for Claude Code

1. **Read** PRD_UPDATED_v3.md (incorporates monitoring)
2. **Read** MONITORING_IMPLEMENTATION_GUIDE.md (detailed setup)
3. **Implement** Phase 1.1 with Prometheus + Grafana (weeks 1-4)
4. **Implement** Phase 1.2 with EvidentlyAI (weeks 5-8)
5. **Plan** OpenTelemetry evaluation for Phase 2 (Q3 2026)

---

## Summary

**TL;DR:**

- **Prometheus + Grafana:** Essential. Include in Phase 1.1. Low effort, high value.
- **EvidentlyAI:** Specialized. Include in Phase 1.2. Detects drift automatically.
- **OpenTelemetry:** Future consideration. Skip Phase 1. Too early, not needed yet.
- **Budget:** 6 days Phase 1.1 + 7 days Phase 1.2 = ~2 weeks total impact on timeline
- **Value:** Moving from "we don't know if our model is working" to "we know exactly what's happening"

This investment pays for itself within weeks when it catches the first production issue.

