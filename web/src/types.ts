export type Outcome = "TP" | "TN" | "FP" | "FN";

export interface Example {
  id: string;
  kepid: number;
  label: 0 | 1;
  outcome: Outcome;
  score: number;
  prob_calibrated: number;
  predicted_candidate: boolean;
  period_days: number;
  duration_hours: number;
  depth_ppm: number;
  snr: number;
  report: string;
  report_source: string;
  light_curve: { t: number[]; f: number[] };
  transits: number[];
  global_view: number[];
  local_view: number[];
}

export interface MeanStd {
  mean: number;
  std: number | null;
  values?: number[];
}

export type MetricName = "roc_auc" | "pr_auc" | "precision_at_recall90";
export type MetricSet = Record<MetricName, MeanStd>;
export interface PairedDiff {
  mean: number;
  std: number | null;
  n_seeds_better: number;
}

export interface BaselineModel {
  test: {
    roc_auc: number;
    pr_auc: number;
    precision_at_recall90: number;
    n: number;
    ci95: Record<MetricName, [number, number]>;
  };
}

export interface BinRow {
  bin: string;
  n: number;
  n_pos: number;
  recall: number | null;
  precision: number | null;
  fpr: number | null;
}

export interface RobustnessEntry {
  n_kept_mean: number;
  n_test_mean: number;
  pr_auc: MeanStd;
  recall: MeanStd;
  precision: MeanStd;
  pr_auc_vs_original: MeanStd;
  recall_vs_original: MeanStd;
}

export interface VettingSummary {
  n: number;
  n_llm: number;
  fallback_rate: number;
  flagged_rate: number;
  flag_rates: Record<string, number>;
  seconds_median: number;
  seconds_p95: number;
  words_mean: number;
}

export interface Results {
  kepler_dataset: {
    n_examples: number;
    n_stars: number;
    examples_per_class: { positivi: number; falsi_positivi: number };
    stars_per_class: { positivi: number; falsi_positivi: number };
    n_stars_with_multiple_kois: number;
  };
  baselines: { splits: Record<string, { n: number; n_stars: number; n_positives: number }>; models: Record<string, BaselineModel> };
  cnn: {
    n_seeds: number;
    summary: Record<string, { metrics: MetricSet; paired_diff_vs_rf: Record<MetricName, PairedDiff> }>;
    baseline_random_forest: MetricSet;
    test_sizes: number[];
  };
  regimes: {
    n_rows: number;
    n_unique_kois: number;
    tables: Record<string, BinRow[]>;
    calibration: Record<string, MeanStd>;
  };
  injection: {
    depths_ppm: number[];
    periods: number[];
    n_injections: number;
    n_valid: number;
    cells: { depth_ppm: number; period: number; mean: number | null; std: number | null; n: number | null }[];
  };
  robustness: Record<string, RobustnessEntry>;
  tess?: {
    dataset: {
      n_examples: number;
      n_stars: number;
      examples_per_class: { positivi: number; falsi_positivi: number };
    };
    n_seeds: number;
    test_sizes: number[];
    summary: Record<string, { metrics: MetricSet; paired_diff: Record<string, Record<MetricName, PairedDiff>> }>;
    kepler_as_is_all_tess: { n: number; prevalence: number; roc_auc_range: [number, number]; pr_auc_range: [number, number] };
  };
  vetting?: {
    n_candidates: number;
    prompt_v1: Record<string, VettingSummary>;
    prompt_v2: Record<string, VettingSummary>;
  };
}

export interface Demo {
  meta: {
    data_kind: string;
    model: string;
    n_seeds: number;
    threshold_score: number;
    n_examples_shown: number;
    shown_by_outcome: Record<Outcome, number>;
    test_set_size: number;
    dataset_examples: number;
    report_model: string | null;
    n_template_reports: number;
    torch: string;
  };
  results: Results;
  examples: Example[];
}
