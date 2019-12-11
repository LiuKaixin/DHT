//Contributors: Sibo Wang, Renchi Yang
#ifndef __ALGO_H__
#define __ALGO_H__

#include "graph.h"
#include "heap.h"
#include "config.h"
#include <tuple>
// #include "sfmt/SFMT.h"
#include <queue>

using namespace std;


struct PredResult {
    double topk_avg_relative_err;
    double topk_avg_abs_err;
    double topk_recall;
    double topk_precision;
    int real_topk_source_count;

    PredResult(double mae = 0, double mre = 0, double rec = 0, double pre = 0, int count = 0) :
            topk_avg_relative_err(mae),
            topk_avg_abs_err(mre),
            topk_recall(rec),
            topk_precision(pre),
            real_topk_source_count(count) {}
};

unordered_map<int, PredResult> pred_results;

Fwdidx fwd_idx;
Bwdidx bwd_idx;
iMap<double> ppr;

iMap<double> ppr_bi;
iMap<double> dht;

iMap<int> topk_filter;
// vector< boost::atomic<double> > vec_ppr;
iMap<int> rw_counter;

iMap<double> rw_bippr_counter;
// RwIdx rw_idx;
atomic<unsigned long long> num_hit_idx;
atomic<unsigned long long> num_total_rw;
atomic<unsigned long long> num_total_bi;
atomic<unsigned long long> num_total_fo;
long num_iter_topk;
vector<int> rw_idx;
vector<pair<unsigned long long, unsigned long> > rw_idx_info;

map<int, vector<pair<int, double> > > exact_topk_pprs;
map<int, vector<pair<int, double>>> exact_topk_dhts;
vector<pair<int, double> > topk_pprs;
vector<pair<int, double> > topk_dhts;

iMap<double> upper_bounds;
iMap<double> lower_bounds;

iMap<double> upper_bounds_self;
iMap<double> upper_bounds_self_init;
iMap<double> lower_bounds_self;

iMap<double> upper_bounds_dht;
iMap<double> lower_bounds_dht;

unordered_map<int, double> multi_bwd_idx_p;
unordered_map<int, unordered_map<int, double>> multi_bwd_idx_r;


vector<pair<int, double>> map_lower_bounds;

// for hubppr
vector<int> hub_fwd_idx;
//pointers to compressed fwd_idx, nodeid:{ start-pointer, start-pointer, start-pointer,...,end-pointer }
map<int, vector<unsigned long long> > hub_fwd_idx_cp_pointers;
vector<vector<unsigned long long>> hub_fwd_idx_ptrs;
vector<int> hub_fwd_idx_size;
iMap<int> hub_fwd_idx_size_k;
vector<int> hub_sample_number;
iMap<int> hub_counter;

map<int, vector<HubBwdidxWithResidual>> hub_bwd_idx;

unsigned concurrency;

vector<int> ks;

vector<unordered_map<int, double>> residual_maps;
vector<map<int, double>> reserve_maps;

inline uint32_t xor128(void) {
    static uint32_t x = 123456789;
    static uint32_t y = 362436069;
    static uint32_t z = 521288629;
    static uint32_t w = 88675123;
    uint32_t t;
    t = x ^ (x << 11);
    x = y;
    y = z;
    z = w;
    return w = w ^ (w >> 19) ^ (t ^ (t >> 8));
}

inline static unsigned long xshift_lrand() {
    return (unsigned long) xor128();
}

inline static double xshift_drand() {
    return ((double) xshift_lrand() / (double) UINT_MAX);
}

inline static unsigned long lrand() {
    return rand();
    // return sfmt_genrand_uint32(&sfmtSeed);
}

inline static double drand() {
    return rand() * 1.0f / RAND_MAX;
    // return sfmt_genrand_real1(&sfmtSeed);
}

inline int random_walk(int start, const Graph &graph) {
    int cur = start;
    unsigned long k;
    if (graph.g[start].size() == 0) {
        return start;
    }
    while (true) {
        if (drand() < config.alpha) {
            return cur;
        }
        if (graph.g[cur].size()) {
            k = lrand() % graph.g[cur].size();
            cur = graph.g[cur][k];
        } else {
            cur = start;
        }
    }
}

inline void random_walk_dht(int start, int pathid, const Graph &graph, unordered_map<int, pair<int, int>> &occur) {
    int cur = start;
    unsigned long k;
    bool flag = true;
    if (graph.g[start].size() == 0) {
        //return start;
        flag = false;
        if (occur.find(cur) == occur.end()) {
            occur.emplace(cur, make_pair(pathid, 1));
        } else if (occur.at(cur).first != pathid) {
            occur.at(cur).first = pathid;
            occur.at(cur).second++;
        }
    }
    while (flag) {
        if (occur.find(cur) == occur.end()) {
            occur.emplace(cur, make_pair(pathid, 1));
        } else if (occur.at(cur).first != pathid) {
            occur.at(cur).first = pathid;
            occur.at(cur).second++;
        }
        if (drand() < config.alpha) {
            //return cur;
            break;
        }
        if (graph.g[cur].size()) {
            k = lrand() % graph.g[cur].size();
            cur = graph.g[cur][k];
        } else {
            cur = start;
        }
    }
}

void global_iteration(int v, vector<double> &dht_old, const Graph &graph) {
    vector<double> new_dht;
    /*
    for(auto item:dht_old){
        int nodeid=item.first;
        for(int nei:graph.gr[nodeid]){
            if (nei==v)continue;
            int deg=graph.g[nei].size();
            new_dht[nei] += (1-config.alpha)/deg* item.second;
        }
    }*/
    double max_dht = 0;
    for (int i = 0; i < graph.n; ++i) {
        if (i == v)continue;
        int deg = graph.g[i].size();
        for (int nei:graph.g[i]) {
            new_dht[i] += (1 - config.alpha) / deg * dht_old.at(nei);
        }
    }
    new_dht[v] += 1;
    swap(dht_old, new_dht);
}

inline double compute_dne_error(double max_error, int k, double max_gap_error) {

    double lambda = pow((1 - config.alpha), k) / config.alpha * max_gap_error;
    return (1 - config.alpha) * (1 - config.alpha) / (config.alpha * (2 - config.alpha)) * (max_error + lambda) +
           lambda;
}

inline bool if_dne_stop(double &old_error, double max_error, int k, double max_gap_error) {
    double new_error = compute_dne_error(max_error, k, max_gap_error);
    cout << new_error << " " << old_error << " " << config.epsilon * config.delta << endl;
    if (new_error <= config.epsilon * config.delta) return true;
    if (old_error >= new_error && old_error / new_error < 1.0 + pow(0.1, 10)) return true;
    old_error = new_error;
    return false;
}

double dhe_query_basic(int query_node, int v, int den_m, const Graph &graph) {
    unordered_map<int, double> dht_to_v, dht_to_v_copy;
    dht_to_v[v] = 1;
    set<int> neighbors, boundary;
    neighbors.emplace(v);
    boundary.emplace(v);
    int max_node = -1;
    double max_error = 0, factor_m =
            (1 - config.alpha) * (1 - config.alpha) / (config.alpha * (2 - config.alpha));//\tau^2/(1-\tau^2)
    //while (neighbors.size() < den_m && !boundary.empty()) {
    while (!boundary.empty()) {
        if (neighbors.size() % 100 == 99) {
            INFO(max_error);
            INFO(neighbors.size());
        }
        max_error = 0;
        for (int node_in_bound:boundary) {
            if (dht_to_v.at(node_in_bound) > max_error) {
                max_error = dht_to_v.at(node_in_bound);
                max_node = node_in_bound;
            }
        }
        if (max_error * factor_m < config.epsilon * config.delta) {
            INFO(max_error);
            INFO(neighbors.size());
            break;
        }
        boundary.erase(max_node);
        for (int nei:graph.gr[max_node]) {
            neighbors.emplace(nei);
        }
        for (int nei:graph.gr[max_node]) {
            for (int nei_of_nei:graph.gr[nei]) {
                if (neighbors.find(nei_of_nei) == neighbors.end()) {
                    boundary.emplace(nei);
                    break;
                }
            }
        }
        for (int node:neighbors) {
            if (node == v) {
                dht_to_v[node] = 1;
                continue;
            } else {
                dht_to_v[node] = 0;
            }
            int deg = graph.g[node].size();
            for (int nei:graph.g[node]) {
                dht_to_v[node] += (1 - config.alpha) / deg * dht_to_v[nei];
            }
        }
    }
    //refinement
    Timer tm(111);
    max_error = 0;
    int k = 0;
    double old_error = 0, max_gap_first = 0;
    do {
        k++;
        for (int node:neighbors) {
            if (node == v) {
                dht_to_v_copy[node] = 1;
                continue;
            } else {
                dht_to_v_copy[node] = 0;
            }
            int deg = graph.g[node].size();
            for (int nei:graph.g[node]) {
                dht_to_v_copy[node] += (1 - config.alpha) / deg * dht_to_v[nei];
            }
            if (boundary.find(node) != boundary.end() && max_error < dht_to_v_copy[node]) {
                max_error = dht_to_v_copy[node];
            }
        }
        if (k == 1) {
            for (auto item:dht_to_v_copy) {
                if (max_gap_first < item.second - dht_to_v[item.first]) {
                    max_gap_first = item.second - dht_to_v[item.first];
                }
            }
        }
        swap(dht_to_v, dht_to_v_copy);
    } while (!if_dne_stop(old_error, max_error, k, max_gap_first));
    return dht_to_v[query_node];
}


unsigned int SEED = 1;

inline static unsigned long lrand_thd(int core_id) {
    //static thread_local std::mt19937 gen(core_id+1);
    //static std::uniform_int_distribution<> dis(0, INT_MAX);
    //return dis(gen);
    return rand_r(&SEED);
}

inline static double drand_thd(int core_id) {
    return ((double) lrand_thd(core_id) / (double) INT_MAX);
}

inline int random_walk_thd(int start, const Graph &graph, int core_id) {
    int cur = start;
    unsigned long k;
    if (graph.g[start].size() == 0) {
        return start;
    }
    while (true) {
        if (drand_thd(core_id) < config.alpha) {
            return cur;
        }
        if (graph.g[cur].size()) {
            k = lrand_thd(core_id) % graph.g[cur].size();
            cur = graph.g[cur][k];
        } else {
            cur = start;
        }
    }
}

void count_hub_dest() {
    // {   Timer tm(101);
    int remaining;
    unsigned long long last_beg_ptr;
    unsigned long long end_ptr;
    int hub_node;
    int blocked_num;
    int bit_pos;
    for (int i = 0; i < hub_counter.occur.m_num; i++) {
        hub_node = hub_counter.occur[i];
        last_beg_ptr = hub_fwd_idx_ptrs[hub_node][hub_fwd_idx_ptrs[hub_node].size() - 2];
        end_ptr = hub_fwd_idx_ptrs[hub_node][hub_fwd_idx_ptrs[hub_node].size() - 1];

        remaining = hub_counter[hub_node];

        if (remaining > hub_fwd_idx_size_k[hub_node]) {
            for (unsigned long long ptr = last_beg_ptr; ptr < end_ptr; ptr += 2) {
                if (rw_counter.notexist(hub_fwd_idx[ptr])) {
                    rw_counter.insert(hub_fwd_idx[ptr], hub_fwd_idx[ptr + 1]);
                } else {
                    rw_counter[hub_fwd_idx[ptr]] += hub_fwd_idx[ptr + 1];
                }
                remaining -= hub_fwd_idx[ptr + 1];
            }
        }


        for (int j = 0; j < hub_fwd_idx_ptrs[hub_node].size() - 2; j++) {
            bit_pos = 1 << j;
            if (bit_pos & remaining) {
                for (unsigned long long ptr = hub_fwd_idx_ptrs[hub_node][j];
                     ptr < hub_fwd_idx_ptrs[hub_node][j + 1]; ptr += 2) {
                    if (rw_counter.notexist(hub_fwd_idx[ptr])) {
                        rw_counter.insert(hub_fwd_idx[ptr], hub_fwd_idx[ptr + 1]);
                    } else {
                        rw_counter[hub_fwd_idx[ptr]] += hub_fwd_idx[ptr + 1];
                    }
                }
            }
        }
    }
    // }
}

inline int random_walk_with_compressed_forward_oracle(int start, const Graph &graph) {
    int cur = start;
    unsigned long k;
    if (graph.g[start].size() == 0) {
        return start;
    }
    while (true) {
        if (hub_fwd_idx_size[cur] != 0 && (hub_counter.notexist(cur) || hub_counter[cur] < hub_fwd_idx_size[cur])) {
            if (hub_counter.notexist(cur))
                hub_counter.insert(cur, 1);
            else
                hub_counter[cur] += 1;
            return -1;
        }

        if (drand() < config.alpha) {
            return cur;
        }

        if (graph.g[cur].size()) {
            k = lrand() % graph.g[cur].size();
            cur = graph.g[cur][k];
        } else
            cur = start;
    }
}

inline void generate_accumulated_fwd_randwalk(int s, const Graph &graph, unsigned long long num_rw) {
    if (graph.g[s].size() == 0) {
        if (rw_counter.notexist(s)) {
            rw_counter.insert(s, num_rw);
        } else {
            rw_counter[s] += num_rw;
        }
        return;
    }

    if (config.with_rw_idx) {
        if (hub_fwd_idx_size[s] != 0) {
            if (num_rw <= hub_fwd_idx_size[s]) {
                hub_counter.insert(s, num_rw);
                num_rw = 0;
            } else {
                hub_counter.insert(s, hub_fwd_idx_size[s]);
                num_rw -= hub_fwd_idx_size[s];
            }
            for (unsigned long long i = 0; i < num_rw; i++) {
                int t = random_walk_with_compressed_forward_oracle(s, graph);
                if (rw_counter.notexist(t)) {
                    rw_counter.insert(t, 1);
                } else {
                    rw_counter[t] += 1;
                }
            }
        } else {
            rw_counter.insert(s, num_rw * config.alpha);
            // rw_counter[s] += num_rw*config.alpha;
            num_rw = num_rw * (1 - config.alpha);
            for (unsigned long long i = 0; i < num_rw; i++) {
                int v = graph.g[s][lrand() % graph.g[s].size()];
                int t = random_walk_with_compressed_forward_oracle(v, graph);
                if (rw_counter.notexist(t)) {
                    rw_counter.insert(t, 1);
                } else {
                    rw_counter[t] += 1;
                }
            }
        }

        count_hub_dest();
        hub_counter.clean();
    } else {
        for (unsigned long long i = 0; i < num_rw; i++) {
            int t = random_walk(s, graph);
            if (rw_counter.notexist(t)) {
                rw_counter.insert(t, 1);
            } else {
                rw_counter[t] += 1;
            }
        }
    }
}

inline void split_line() {
    INFO("-----------------------------");
}

inline void display_setting() {
    INFO(config.epsilon);
    INFO(config.delta);
    INFO(config.pfail);
    INFO(config.rmax);
    INFO(config.omega);
    if (config2.pfail == config.pfail) {
        INFO(config2.epsilon);
        INFO(config2.delta);
        INFO(config2.pfail);
        INFO(config2.rmax);
        INFO(config2.omega);
    }
}

inline void display_fwdidx() {
    for (int i = 0; i < fwd_idx.first.occur.m_num; i++) {
        int nodeid = fwd_idx.first.occur[i];
        cout << "k:" << nodeid << " v:" << fwd_idx.first[nodeid] << endl;
    }

    cout << "=======================" << endl;

    for (int i = 0; i < fwd_idx.second.occur.m_num; i++) {
        int nodeid = fwd_idx.second.occur[i];
        cout << "k:" << nodeid << " v:" << fwd_idx.second[nodeid] << endl;
    }
}

inline void display_ppr() {
    for (int i = 0; i < ppr.occur.m_num; i++) {
        cout << ppr.occur[i] << "->" << ppr[ppr.occur[i]] << endl;
    }
}

inline void display_dht() {
    for (int i = 0; i < dht.occur.m_num; i++) {
        if (dht[dht.occur[i]] <= 0)continue;
        cout << dht.occur[i] << "->" << dht[dht.occur[i]] << endl;
    }
}

inline void display_topk_dht() {
    for (int i = 0; i < topk_dhts.size(); i++) {
        cout << topk_dhts[i].first << "->" << topk_dhts[i].second << endl;
    }
}

static void display_time_usage(int used_counter, int query_size) {
    if (config.algo == FORA) {
        cout << "Total cost (s): " << Timer::used(used_counter) << endl;
        cout << Timer::used(RONDOM_WALK) * 100.0 / Timer::used(used_counter) << "%" << " for random walk cost" << endl;
        cout << Timer::used(FWD_LU) * 100.0 / Timer::used(used_counter) << "%" << " for forward push cost" << endl;
        // if(config.action == TOPK)
        // cout <<  Timer::used(SORT_MAP)*100.0/Timer::used(used_counter) << "%" << " for sorting top k cost" << endl;
        split_line();
    } else if (config.algo == FORA_MC) {
        cout << "Total cost (s): " << Timer::used(used_counter) << endl;
        cout << Timer::used(RONDOM_WALK) * 100.0 / Timer::used(used_counter) << "%" << " for random walk cost" << endl;
        cout << Timer::used(FWD_LU) * 100.0 / Timer::used(used_counter) << "%" << " for forward push cost" << endl;
        cout << Timer::used(MC_QUERY2) * 100.0 / Timer::used(used_counter) << "%" << " for montecarlo cost" << endl;
        split_line();
    } else if (config.algo == BIPPR) {
        cout << "Total cost (s): " << Timer::used(used_counter) << endl;
        cout << Timer::used(RONDOM_WALK) * 100.0 / Timer::used(used_counter) << "%" << " for random walk cost" << endl;
        cout << Timer::used(BWD_LU) * 100.0 / Timer::used(used_counter) << "%" << " for backward push cost" << endl;
    } else if (config.algo == MC) {
        cout << "Total cost (s): " << Timer::used(used_counter) << endl;
        cout << Timer::used(RONDOM_WALK) * 100.0 / Timer::used(used_counter) << "%" << " for random walk cost" << endl;
    } else if (config.algo == FWDPUSH) {
        cout << "Total cost (s): " << Timer::used(used_counter) << endl;
        cout << Timer::used(FWD_LU) * 100.0 / Timer::used(used_counter) << "%" << " for forward push cost" << endl;
    } else if (config.algo == MC_DHT) {
        cout << "Total cost (s): " << Timer::used(used_counter) << endl;
        cout << Timer::used(RONDOM_WALK) * 100.0 / Timer::used(used_counter) << "%" << " for random walk cost" << endl;
    }

    if (config.with_rw_idx)
        cout << "Average rand-walk idx hit ratio: " << num_hit_idx * 100.0 / num_total_rw << "%" << endl;

    if (config.action == TOPK) {
        assert(result.real_topk_source_count > 0);
        cout << "Average top-K Precision: " << result.topk_precision / result.real_topk_source_count << endl;
        cout << "Average top-K Recall: " << result.topk_recall / result.real_topk_source_count << endl;
    }

    cout << "Average query time (s):" << Timer::used(used_counter) / query_size << endl;
    cout << "Memory usage (MB):" << get_proc_memory() / 1000.0 << endl << endl;
}

static void set_result(const Graph &graph, int used_counter, int query_size) {
    config.query_size = query_size;

    result.m = graph.m;
    result.n = graph.n;
    result.avg_query_time = Timer::used(used_counter) / query_size;

    result.total_mem_usage = get_proc_memory() / 1000.0;
    result.total_time_usage = Timer::used(used_counter);

    result.num_randwalk = num_total_rw;

    if (config.with_rw_idx) {
        result.num_rw_idx_use = num_hit_idx;
        result.hit_idx_ratio = num_hit_idx * 1.0 / num_total_rw;
    }

    result.randwalk_time = Timer::used(RONDOM_WALK);
    result.randwalk_time_ratio = Timer::used(RONDOM_WALK) * 100 / Timer::used(used_counter);

    if (config.algo == FORA) {
        result.propagation_time = Timer::used(FWD_LU);
        result.propagation_time_ratio = Timer::used(FWD_LU) * 100 / Timer::used(used_counter);
        // result.source_dist_time = Timer::used(SOURCE_DIST);
        // result.source_dist_time_ratio = Timer::used(SOURCE_DIST)*100/Timer::used(used_counter);
    } else if (config.algo == BIPPR) {
        result.propagation_time = Timer::used(BWD_LU);
        result.propagation_time_ratio = Timer::used(BWD_LU) * 100 / Timer::used(used_counter);
    } else if (config.algo == FB) {
        result.propagation_time = Timer::used(FWD_LU) + Timer::used(BWD_LU);
        result.propagation_time_ratio = result.propagation_time * 100 / Timer::used(used_counter);
    }

    if (config.action == TOPK) {
        result.topk_sort_time = Timer::used(SORT_MAP);
        // result.topk_precision = avg_topk_precision;
        // result.topk_sort_time_ratio = Timer::used(SORT_MAP)*100/Timer::used(used_counter);
    }
}

inline void bippr_setting(int n, long long m) {
    config.rmax = config.epsilon * sqrt(m * 1.0 * config.delta / 3.0 / log(2.0 / config.pfail));
    // config.omega = m/config.rmax;
    config.rmax *= config.rmax_scale;
    config.omega = config.rmax * 3 * log(2.0 / config.pfail) / config.delta / config.epsilon / config.epsilon;
}

inline void bippr_setting_lkx(int n, long long m) {
    config2.rmax = config2.epsilon * sqrt(m * 1.0 * config2.delta / 3.0 / n / log(2.0 / config2.pfail));
    // config.omega = m/config.rmax;
    config2.rmax *= config2.rmax_scale;
    config2.omega = config2.rmax * 3 * log(2.0 / config2.pfail) / config2.delta / config2.epsilon / config2.epsilon;
}

inline void hubppr_topk_setting(int n, long long m) {
    config.rmax = config.epsilon * sqrt(m * 1.0 * config.delta / 3.0 / log(2.0 / config.pfail));
    config.rmax *= config.rmax_scale;
    config.omega = config.rmax * 3 * log(2.0 / config.pfail) / config.delta / config.epsilon / config.epsilon;
}

inline void fora_setting(int n, long long m) {
    config.rmax = config.epsilon * sqrt(config.delta / 3 / m / log(2 / config.pfail));
    config.rmax *= config.rmax_scale;
    // config.rmax *= config.multithread_param;
    config.omega = (2 + config.epsilon) * log(2 / config.pfail) / config.delta / config.epsilon / config.epsilon;
}

inline void fb_raw_setting(int n, long long m, double ratio, double raw_epsilon) {
    config.pfail = 1.0 / 2 / n;
    config2 = config;

    config.delta = config.alpha / n;
    config2.delta = config2.alpha;
    config.epsilon = raw_epsilon * ratio / (ratio + 1 + raw_epsilon);
    config2.epsilon = raw_epsilon / (ratio + 1 + raw_epsilon);

    config.rmax = config.epsilon * sqrt(config.delta / 3 / m / log(2 / config.pfail));
    config.rmax *= config.rmax_scale;
    config.omega = (2 + config.epsilon) * log(2 / config.pfail) / config.delta / config.epsilon / config.epsilon;

    config2.rmax = config2.epsilon * sqrt(config2.delta / 3 / log(2.0 / config2.pfail));
    config2.rmax *= config2.rmax_scale;
    config2.omega = config2.rmax * (2 + config2.epsilon) * log(2.0 / config2.pfail) / config2.delta / config2.epsilon /
                    config2.epsilon;
}

bool compare_fora_bippr_cost(int k, int m = 0, double rsum = 0) {
    if (k == 0)
        return false;
    static double old_config2_rmax;
    //比较前向和后向的复杂度，如果前向高于后向，返回真
    double ratio;
    double cost_fora =
            (2 * config.epsilon / 3 + 2) * log(2 / config.pfail) / config.delta / config.epsilon / config.epsilon;
    INFO(cost_fora);
    INFO(1 / config.rmax + rsum * config.omega);
    INFO(k * (2 / config2.rmax));
    //ratio=(1/config.rmax+rsum*config.omega)/k/(2/config2.rmax);
    ratio = ((config.rmax * m + rsum) * cost_fora * (sqrt(2) - 1) / k / (2 / config2.rmax));
    old_config2_rmax = config2.rmax;
    return ratio > 1;
}

bool test_run_fora(int candidate_size, long m, int n, double old_f_rmax, double old_b_rmax, double rsum = 0) {
    if (candidate_size == 0)
        return true;

    double factor_rmax = (log(m / n) / log((1 - config.alpha) / m * n) - 1);
    double product_rmax = log(
            m * (2 * config.epsilon / 3 + 2) * log(2 / config.pfail) / config.epsilon / config.epsilon / (config.delta/2));
    double rmax_new = exp(product_rmax / factor_rmax);
    double k_new = log((m/n) * rmax_new) / log((1 - config.alpha) / (m/n));
    double k = log((m/n) * config.rmax) / log((1 - config.alpha) / (m/n));
    double cost_fp=pow((m/n),k_new)-pow((m/n),k);
    /*
   double cost_fp;
   if (config.rmax * m < 1) {
       double threshold_k = log(n) / log(m / n);
       double k = log(m * config.rmax) / log(1 - config.alpha);
       cost_fp = (k - threshold_k - 1) * m;
   } else {
       double k = log(m / n * config.rmax) / log((1 - config.alpha) / m * n);
       cost_fp = m / n / (m / n - 1) * pow(m / n, k);
   }*/
    //double cost_fp=m*(1+log(m*config.rmax)/log(1-config.alpha));
    //先尝试对第一步骤进行
    double cost_rw_fora = fwd_idx.second.occur.m_size - fwd_idx.first.occur.m_size;
    if (cost_rw_fora < rsum * config.omega) {
        cost_rw_fora = rsum * config.omega;
    }
    double cost_bippr = candidate_size / config2.epsilon * sqrt((m / n) * 3 * log(2 / config2.pfail) / config2.delta);
    INFO(cost_fp, cost_rw_fora, cost_bippr);
    return cost_fp + cost_rw_fora < cost_bippr;
    /*
    double cost_foward_push = log((m/n) * config.rmax) / log((1 - config.alpha) / (m/n)) + 1;
    cost_foward_push = pow((m/n), cost_foward_push);

    double cost_bippr = candidate_size*() / config2.epsilon * sqrt((m/n) * 3 * log(2 / config2.pfail) / config2.delta);
    INFO(cost_foward_push, cost_rw_fora, cost_bippr);
    return cost_foward_push + cost_rw_fora < cost_bippr;*/

}

inline void fora_bippr_setting(int n, long long m, double ratio, double raw_epsilon, bool topk = false) {
    if (!topk) {
        config.pfail = 1.0 / 2 / n;
        config.delta = config.alpha / n;
        config2 = config;

        config.epsilon = raw_epsilon * ratio / (ratio + 1 + raw_epsilon);
        config2.epsilon = raw_epsilon / (ratio + 1 + raw_epsilon);

        config2.delta = config2.alpha;
    }
    /*
    double deg = m * 1.0 / n;
    double factor_fp=log(deg)/log((1-config.alpha)/deg);
    double value =
            m * 3 * log(2 / config.pfail) / config.epsilon / config.epsilon / config.delta;
    config.rmax=log(-1*value/factor_fp*(deg-1)/deg)/(factor_fp-1);
    config.rmax=exp(config.rmax)/deg;
    INFO(value,factor_fp);

    if (config.rmax * m < 1) {
        config.rmax = config.epsilon * config.epsilon * config.delta / 3 /log(2 / config.pfail);
        INFO(config.rmax * m);
    }
    */
    double factor_rmax = (log(m / n) / log((1 - config.alpha) / m * n) - 1);
    double product_rmax = log(
            m * (2 * config.epsilon / 3 + 2) * log(2 / config.pfail) / config.epsilon / config.epsilon / config.delta);
    double product_rmax2= -1*log(m / n)*log(m / n)/log((1-config.alpha)/(m / n))-log(m / n)/log(m / n-1);
    config.rmax = exp((product_rmax+product_rmax2) / factor_rmax);
    //config.rmax= config.epsilon*config.epsilon*config.delta/m/3/log(2/config.pfail)/(-1*log(1-config.alpha));
    //config.rmax = config.epsilon * sqrt(config.delta / 3 / m / log(2 / config.pfail));
    //config.rmax = config.epsilon * sqrt(config.delta / 3 / n / log(2 / config.pfail));
    //config.rmax =config.epsilon / n * sqrt(config.delta * m / 3 / config.alpha / log(2 / config.pfail));
    //config.rmax = config.epsilon / n * sqrt(config.delta * m / 3  / log(2 / config.pfail));
    //config.rmax = config.epsilon * sqrt(config.delta / 3 / m / log(2 / config.pfail));
    //config.rmax = config.epsilon * sqrt(config.delta / 3 / m / log(2 / config.pfail)/7);
    config.rmax *= config.rmax_scale;
    // config.rmax *= config.multithread_param;
    config.omega = (2 + config.epsilon) * log(2 / config.pfail) / config.delta / config.epsilon / config.epsilon;


    config2.rmax = config2.epsilon * sqrt(config2.delta * m / n / config.alpha / 3 / log(2.0 / config2.pfail));
    //config2.rmax = config2.epsilon * sqrt(m * config2.delta / 3.0 / n / log(2.0 / config2.pfail));
    //config2.rmax = config2.epsilon * sqrt( m * config2.delta / 3.0 / n / log(2.0 / config2.pfail)/8);
    // config.omega = m/config.rmax;
    config2.rmax *= config2.rmax_scale;
    config2.omega = config2.rmax * (2 + config2.epsilon) * log(2.0 / config2.pfail) / config2.delta / config2.epsilon /
                    config2.epsilon;
}

/*
inline void fora_bippr_setting(int n, long long m, double ratio, double raw_epsilon, bool topk = false) {
    if (!topk) {
        config.pfail = 1.0 / 2 / n;
        config.delta = config.alpha / n;
        config2 = config;

        config.epsilon = raw_epsilon * ratio / (ratio + 1 + raw_epsilon);
        config2.epsilon = raw_epsilon / (ratio + 1 + raw_epsilon);

        config2.delta = config2.alpha;
    }
    double factor_rmax=(1/log((1-config.alpha)/m*n)-1);
    double product_rmax=log(m*(2*config.epsilon/3+2)*log(2/config.pfail)/config.epsilon/config.epsilon/config.delta);
    double rmax=exp(product_rmax/factor_rmax);
    config.rmax = config.epsilon * sqrt(config.delta / 3 / m / log(2 / config.pfail));
    //config.rmax = config.epsilon * sqrt(config.delta / 3 / n / log(2 / config.pfail));
    //config.rmax =config.epsilon / n * sqrt(config.delta * m / 3 / config.alpha / log(2 / config.pfail));
    //config.rmax = config.epsilon / n * sqrt(config.delta * m / 3  / log(2 / config.pfail));
    //config.rmax = config.epsilon * sqrt(config.delta / 3 / m / log(2 / config.pfail));
    //config.rmax = config.epsilon * sqrt(config.delta / 3 / m / log(2 / config.pfail)/7);
    config.rmax *= config.rmax_scale;
    // config.rmax *= config.multithread_param;
    config.omega = (2 + config.epsilon) * log(2 / config.pfail) / config.delta / config.epsilon / config.epsilon;
    config2.rmax = config2.epsilon * sqrt(config2.delta / 3 / log(2.0 / config2.pfail));
    //config2.rmax = config2.epsilon * sqrt(m * config2.delta / 3.0 / n / log(2.0 / config2.pfail));
    //config2.rmax = config2.epsilon * sqrt( m * config2.delta / 3.0 / n / log(2.0 / config2.pfail)/8);
    // config.omega = m/config.rmax;
    config2.rmax *= config2.rmax_scale;
    config2.omega = config2.rmax * (2 + config2.epsilon) * log(2.0 / config2.pfail) / config2.delta / config2.epsilon /
                    config2.epsilon;
}
*/
inline void montecarlo_setting() {
    double fwd_rw_count = 3 * log(2 / config.pfail) / config.epsilon / config.epsilon / config.delta;
    config.omega = fwd_rw_count;
}

inline void montecarlo_dht_setting() {
    double fwd_rw_count = 3 * log(2 / config.pfail) / config.epsilon / config.epsilon / config.delta;
    config.omega = fwd_rw_count;
}

inline void montecarlo_setting2() {
    double fwd_rw_count = 3 * log(2 / config.pfail) / config.epsilon / config.epsilon / config.alpha;
    config.omega = fwd_rw_count;
}

inline void fwdpush_setting(int n, long long m) {
    // below is just a estimate value, has no accuracy guarantee
    // since for undirected graph, error |ppr(s, t)-approx_ppr(s, t)| = sum( r(s, v)*ppr(v, t)) <= d(t)*rmax
    // |ppr(s, t)-approx_ppr(s, t)| <= epsilon*ppr(s, t)
    // d(t)*rmax <= epsilon*ppr(s, t)
    // rmax <= epsilon*ppr(s, t)/d(t)
    // d(t): use average degree d=m/n
    // ppr(s, t): use minimum ppr value, delta, i.e., 1/n
    // thus, rmax <= epsilon*delta*n/m = epsilon/m
    // use config.rmax_scale to tune rmax manually
    config.rmax = config.rmax_scale * config.delta * config.epsilon * n / m;
}

/*

    vector<int> queries;
    load_ss_query(queries);
    INFO(queries.size());
    unsigned int query_size = queries.size();
    query_size = min(query_size, config.query_size);
    int used_counter = 0;

    assert(config.k < graph.n - 1);
    assert(config.k > 1);
    INFO(config.k);

    split_line();

    load_exact_topk_ppr();

    // not FORA, so it's single source
    // no need to change k to run again
    // check top-k results for different k
    if (config.algo != FORA && config.algo != HUBPPR) {
        unsigned int step = config.k / 5;
        if (step > 0) {
            for (unsigned int i = 1; i < 5; i++) {
                ks.push_back(i * step);
            }
        }
        ks.push_back(config.k);
        for (auto k: ks) {
            PredResult rst(0, 0, 0, 0, 0);
            pred_results.insert(MP(k, rst));
        }
    }

    used_counter = 0;
    if (config.algo == FORA) {
        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);
        rw_counter.init_keys(graph.n);
        upper_bounds.init_keys(graph.n);
        lower_bounds.init_keys(graph.n);
        ppr.initialize(graph.n);
        topk_filter.initialize(graph.n);
    } else if (config.algo == MC) {
        rw_counter.initialize(graph.n);
        ppr.initialize(graph.n);
        montecarlo_setting();
    } else if (config.algo == BIPPR) {
        bippr_setting(graph.n, graph.m);
        rw_counter.initialize(graph.n);
        bwd_idx.first.initialize(graph.n);
        bwd_idx.second.initialize(graph.n);
        ppr.initialize(graph.n);
    } else if (config.algo == FWDPUSH) {
        fwdpush_setting(graph.n, graph.m);
        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);
        ppr.initialize(graph.n);
    } else if (config.algo == HUBPPR) {
        hubppr_topk_setting(graph.n, graph.m);
        rw_counter.initialize(graph.n);
        upper_bounds.init_keys(graph.n);
        if (config.with_rw_idx) {
            hub_counter.initialize(graph.n);
            load_hubppr_oracle(graph);
        }
        residual_maps.resize(graph.n);
        reserve_maps.resize(graph.n);
        map_lower_bounds.resize(graph.n);
        for (int v = 0; v < graph.n; v++) {
            residual_maps[v][v] = 1.0;
            map_lower_bounds[v] = MP(v, 0);
        }
        updated_pprs.initialize(graph.n);
    }

    for (int i = 0; i < query_size; i++) {
        cout << i + 1 << ". source node:" << queries[i] << endl;
        get_topk(queries[i], graph);
        split_line();
    }

    cout << "average iter times:" << num_iter_topk / query_size << endl;
    display_time_usage(used_counter, query_size);
    set_result(graph, used_counter, query_size);

    //not FORA, so it's single source
    //no need to change k to run again
    // check top-k results for different k
    if (config.algo != FORA && config.algo != HUBPPR) {
        display_precision_for_dif_k();
    }montecarlo_query_dht(queries[i], graph);
            fb_raw_query(queries[i], graph);
            dne_query(queries[i], graph);
            fora_bippr_query(queries[i], graph, raw_epsilon);
{
        Timer timer(RONDOM_WALK); //both rand-walk and source distribution are related with num_random_walk
        { //rand walk online
            for (long i = 0; i < fwd_idx.second.occur.m_num; i++) {
                int source = fwd_idx.second.occur[i];
                double residual = fwd_idx.second[source];
                unsigned long num_s_rw = ceil(residual / check_rsum * num_random_walk);
                double a_s = residual / check_rsum * num_random_walk / num_s_rw;

                double ppr_incre = a_s * check_rsum / num_random_walk;

                num_total_rw += num_s_rw;
                for (unsigned long j = 0; j < num_s_rw; j++) {
                    int des = random_walk(source, graph);
                    ppr[des] += ppr_incre;
                }
            }
        }
    }
 */
inline bool check_cost(double rsum, double &ratio, double n, double m) {
    double cost_bippr = fwd_idx.second.occur.m_num * (config2.omega * 7 + m / n / config2.rmax);//config.alpha;
    double cost_fora = m / n / config.rmax + rsum * config.omega * 7;
    double cost_fora2 = m / n / config.rmax + n * config.rmax * config.omega * 7;
    //double cost_bippr=fwd_idx.second.occur.m_num*config2.omega*(1+1/config2.alpha);
    //double cost_fora=1/config.rmax+rsum*config.omega/config.alpha;
    //double cost_bippr=fwd_idx.second.occur.m_num*config2.omega;
    //double cost_fora=rsum*config.omega;
    cout << "cost:\t" << cost_fora << "\t" << cost_fora2 << "\t" << cost_bippr << endl;
    cout << "rsum:\t" << rsum << "\t" << n * config.rmax << endl;
    cout << fwd_idx.second.occur.m_num << endl;
    if (cost_bippr > cost_fora) {
        ratio /= 2;
        return true;
    }
    return false;
}

inline void generate_ss_query(int n) {
    string filename = config.graph_location + "ssquery.txt";
    ofstream queryfile(filename);
    for (int i = 0; i < config.query_size; i++) {
        int v = rand() % n;
        queryfile << v << endl;
    }
}

inline void generate_high_degree_ss_query(int n, vector<vector<int>> &g) {
    string filename = config.graph_location + "high_degree_ssquery.txt";
    ofstream queryfile(filename);
    struct Node {
        int node_id;
        int node_degree;

        bool operator<(const Node &x) const {
            return node_degree > x.node_degree;
        }
    };
    //设置一个最小堆的优先队列，大小为query_size，然后如果下一个比堆顶元素大，则替换。
    priority_queue<Node> queue;
    for (int i = 0; i < n; i++) {
        Node node;
        node.node_id = i;
        node.node_degree = g[i].size();
        if (queue.size() < config.query_size) {
            queue.push(node);
        } else if (node.node_degree > queue.top().node_degree) {
            queue.pop();
            queue.push(node);
        }
    }
    //改变最小堆得输出顺序
    int result[config.query_size];
    for (int i = 0; i < config.query_size; i++) {
        Node n = queue.top();
        queue.pop();
        result[config.query_size - 1 - i] = n.node_id;
    }
    for (int i = 0; i < config.query_size; i++) {
        queryfile << result[i] << endl;
    }
}

void load_ss_query(vector<int> &queries) {
    string filename = config.query_high_degree ? config.graph_location + "high_degree_ssquery.txt" :
                      config.graph_location + "ssquery.txt";
    if (!file_exists_test(filename)) {
        cerr << "query file does not exist, please generate ss query files first" << endl;
        exit(0);
    }
    ifstream queryfile(filename);
    int v;
    while (queryfile >> v) {
        queries.push_back(v);
    }
}

void compute_precision(int v) {
    double precision = 0.0;
    double recall = 0.0;

    if (exact_topk_pprs.size() > 1 && exact_topk_pprs.find(v) != exact_topk_pprs.end()) {

        unordered_map<int, double> topk_map;
        for (auto &p: topk_pprs) {
            if (p.second > 0) {
                topk_map.insert(p);
            }
        }

        unordered_map<int, double> exact_map;
        int size_e = min(config.k, (unsigned int) exact_topk_pprs[v].size());

        for (int i = 0; i < size_e; i++) {
            pair<int, double> &p = exact_topk_pprs[v][i];
            if (p.second > 0) {
                exact_map.insert(p);
                if (topk_map.find(p.first) != topk_map.end())
                    recall++;
            }
        }

        for (auto &p: topk_map) {
            if (exact_map.find(p.first) != exact_map.end()) {
                precision++;
            }
        }

        // for(int i=0; i<config.k; i++){
        //     cout << "NO." << i << " pred:" << topk_pprs[i].first << ", " << topk_pprs[i].second << "\t exact:" << exact_topk_pprs[v][i].first << ", " << exact_topk_pprs[v][i].second << endl;
        // }

        assert(exact_map.size() > 0);
        assert(topk_map.size() > 0);
        if (exact_map.size() <= 1)
            return;

        recall = recall * 1.0 / exact_map.size();
        precision = precision * 1.0 / exact_map.size();
        INFO(exact_map.size(), recall, precision);
        result.topk_recall += recall;
        result.topk_precision += precision;

        result.real_topk_source_count++;
    }
}

void compute_precision_dht(int v) {
    double precision = 0.0;
    double recall = 0.0;

    if (exact_topk_dhts.size() > 1 && exact_topk_dhts.find(v) != exact_topk_dhts.end()) {

        unordered_map<int, double> topk_map;
        for (auto &p: topk_pprs) {
            if (p.second > 0) {
                topk_map.insert(p);
            }
        }

        unordered_map<int, double> exact_map;
        int size_e = min(config.k, (unsigned int) exact_topk_dhts[v].size());

        for (int i = 0; i < size_e; i++) {
            pair<int, double> &p = exact_topk_dhts[v][i];
            if (p.second > 0) {
                exact_map.insert(p);
                if (topk_map.find(p.first) != topk_map.end())
                    recall++;
            }
        }

        for (auto &p: topk_map) {
            if (exact_map.find(p.first) != exact_map.end()) {
                precision++;
            }
        }

        // for(int i=0; i<config.k; i++){
        //     cout << "NO." << i << " pred:" << topk_pprs[i].first << ", " << topk_pprs[i].second << "\t exact:" << exact_topk_dhts[v][i].first << ", " << exact_topk_dhts[v][i].second << endl;
        // }

        assert(exact_map.size() > 0);
        assert(topk_map.size() > 0);
        if (exact_map.size() <= 1)
            return;

        recall = recall * 1.0 / exact_map.size();
        precision = precision * 1.0 / exact_map.size();
        INFO(exact_map.size(), recall, precision);
        result.topk_recall += recall;
        result.topk_precision += precision;

        result.real_topk_source_count++;
    }
}

inline bool cmp(double x, double y) {
    return x > y;
}

// obtain the top-k ppr values from ppr map
double kth_ppr() {
    Timer tm(SORT_MAP);

    static vector<double> temp_ppr;
    temp_ppr.clear();
    temp_ppr.resize(ppr.occur.m_num);
    int nodeid;
    for (int i; i < ppr.occur.m_num; i++) {
        // if(ppr.m_data[i]>0)
        // temp_ppr[size++] = ppr.m_data[i];
        temp_ppr[i] = ppr[ppr.occur[i]];
    }

    nth_element(temp_ppr.begin(), temp_ppr.begin() + config.k - 1, temp_ppr.end(), cmp);
    return temp_ppr[config.k - 1];
}

double topk_ppr() {
    topk_pprs.clear();
    topk_pprs.resize(config.k);

    static unordered_map<int, double> temp_ppr;
    temp_ppr.clear();
    // temp_ppr.resize(ppr.occur.m_num);
    int nodeid;
    for (long i = 0; i < ppr.occur.m_num; i++) {
        nodeid = ppr.occur[i];
        // INFO(nodeid);
        temp_ppr[nodeid] = ppr[nodeid];
    }

    partial_sort_copy(temp_ppr.begin(), temp_ppr.end(), topk_pprs.begin(), topk_pprs.end(),
                      [](pair<int, double> const &l, pair<int, double> const &r) { return l.second > r.second; });

    return topk_pprs[config.k - 1].second;
}

double topk_dht() {
    topk_dhts.clear();
    topk_dhts.resize(config.k);

    static vector<pair<int, double> > temp_dht;
    temp_dht.clear();
    temp_dht.resize(dht.cur);
    int nodeid, cur = 0;
    for (int k = 0; k < dht.occur.m_num; ++k) {
        nodeid = dht.occur[k];
        if (dht.exist(nodeid))
            temp_dht[cur++] = MP(nodeid, dht[nodeid]);
    }

    partial_sort_copy(temp_dht.begin(), temp_dht.end(), topk_dhts.begin(), topk_dhts.end(),
                      [](pair<int, double> const &l, pair<int, double> const &r) { return l.second > r.second; });

    return topk_dhts[config.k - 1].second;
}

double topk_of(vector<pair<int, double> > &top_k) {
    top_k.clear();
    top_k.resize(config.k);
    static vector<pair<int, double> > temp_ppr;
    temp_ppr.clear();
    temp_ppr.resize(ppr.occur.m_num);
    for (int i = 0; i < ppr.occur.m_num; i++) {
        temp_ppr[i] = MP(ppr.occur[i], ppr[ppr.occur[i]]);
    }

    partial_sort_copy(temp_ppr.begin(), temp_ppr.end(), top_k.begin(), top_k.end(),
                      [](pair<int, double> const &l, pair<int, double> const &r) { return l.second > r.second; });

    return top_k[config.k - 1].second;
}

void compute_precision_for_dif_k(int v) {
    if (exact_topk_pprs.size() > 1 && exact_topk_pprs.find(v) != exact_topk_pprs.end()) {
        //vector<double> true_dht(exact_topk_pprs.size())
        for (auto k: ks) {

            int j = 0;
            unordered_map<int, double> topk_map;
            for (auto &p: topk_pprs) {
                if (p.second > 0) {
                    topk_map.insert(p);
                }
                j++;
                if (j == k) { // only pick topk
                    break;
                }
            }

            double recall = 0.0;
            unordered_map<int, double> exact_map;
            int size_e = min(k, (int) exact_topk_pprs[v].size());
            for (int i = 0; i < size_e; i++) {
                pair<int, double> &p = exact_topk_pprs[v][i];
                if (p.second > 0) {
                    exact_map.insert(p);
                    if (topk_map.find(p.first) != topk_map.end())
                        recall++;
                }
            }

            double precision = 0.0;
            for (auto &p: topk_map) {
                if (exact_map.find(p.first) != exact_map.end()) {
                    precision++;
                }
            }

            if (exact_map.size() <= 1)
                continue;

            precision = precision * 1.0 / exact_map.size();
            recall = recall * 1.0 / exact_map.size();

            pred_results[k].topk_precision += precision;
            pred_results[k].topk_recall += recall;
            pred_results[k].real_topk_source_count++;
        }
    }
}

void compute_precision_for_dif_k_dht(int v) {
    if (exact_topk_dhts.size() > 1 && exact_topk_dhts.find(v) != exact_topk_dhts.end()) {
        //vector<double> true_dht(exact_topk_dhts.size())
        for (auto k: ks) {

            int j = 0;
            unordered_map<int, double> topk_map;
            for (auto &p: topk_pprs) {
                if (p.second > 0) {
                    topk_map.insert(p);
                }
                j++;
                if (j == k) { // only pick topk
                    break;
                }
            }

            double recall = 0.0;
            unordered_map<int, double> exact_map;
            int size_e = min(k, (int) exact_topk_dhts[v].size());
            for (int i = 0; i < size_e; i++) {
                pair<int, double> &p = exact_topk_dhts[v][i];
                if (p.second > 0) {
                    exact_map.insert(p);
                    if (topk_map.find(p.first) != topk_map.end())
                        recall++;
                }
            }

            double precision = 0.0;
            for (auto &p: topk_map) {
                if (exact_map.find(p.first) != exact_map.end()) {
                    precision++;
                }
            }

            if (exact_map.size() <= 1)
                continue;

            precision = precision * 1.0 / exact_map.size();
            recall = recall * 1.0 / exact_map.size();

            pred_results[k].topk_precision += precision;
            pred_results[k].topk_recall += recall;
            pred_results[k].real_topk_source_count++;
        }
    }
}

inline void display_precision_for_dif_k() {
    split_line();
    cout << config.algo << endl;
    for (auto k: ks) {
        cout << k << "\t";
    }
    cout << endl << "Precision:" << endl;
    assert(pred_results[k].real_topk_source_count > 0);
    for (auto k: ks) {
        cout << pred_results[k].topk_precision / pred_results[k].real_topk_source_count << "\t";
    }
    cout << endl << "Recall:" << endl;
    for (auto k: ks) {
        cout << pred_results[k].topk_recall / pred_results[k].real_topk_source_count << "\t";
    }
    cout << endl;
}

inline void init_multi_setting(int n) {
    INFO("multithreading mode...");
    concurrency = std::thread::hardware_concurrency() + 1;
    INFO(concurrency);
    assert(concurrency >= 2);
    config.rmax_scale = sqrt(concurrency * 1.0);
    INFO(config.rmax_scale);
}

int
reverse_local_update_linear_dht(int t, const Graph &graph, vector<int> &idx, vector<int> &node_with_r, int &pointer_r,
                                vector<int> &q, double init_residual = 1) {
    Timer tm(111);
    int backward_counter = 0, pointer_q = 0;
    //vector<int> q;
    //q.reserve(graph.n);
    //q.push_back(-1);
    unsigned long left = 0;

    double myeps = config2.rmax;
    q[pointer_q++] = t;
    //q.push_back(t);
    bwd_idx.second.occur[t] = t;
    bwd_idx.second[t] = 1;
    pointer_r = 0;
    node_with_r[pointer_r++] = t;
    idx[t] = t;
    while (left != pointer_q) {
        int v = q[left];
        idx[v] = -1;
        left++;
        left %= graph.n;
        if (bwd_idx.second[v] < myeps)
            break;

        if (v == t) {
            if (bwd_idx.first.occur[v] != t) {
                bwd_idx.first.occur[v] = t;
                bwd_idx.first[v] = bwd_idx.second[v] * config2.alpha;
            } else
                bwd_idx.first[v] += bwd_idx.second[v] * config.alpha;
        }

        double residual = (1 - config2.alpha) * bwd_idx.second[v];
        bwd_idx.second[v] = 0;
        if (graph.gr[v].size() > 0) {
            backward_counter += graph.gr[v].size();
            for (int next : graph.gr[v]) {
                int cnt = graph.g[next].size();
                if (bwd_idx.second.occur[next] != t) {
                    bwd_idx.second.occur[next] = t;
                    bwd_idx.second[next] = residual / cnt;
                    node_with_r[pointer_r++] = next;
                } else
                    bwd_idx.second[next] += residual / cnt;

                if (bwd_idx.second[next] > myeps && idx[next] != t) {
                    // put next into q if next is not in q
                    idx[next] = t;//(int) q.size();
                    //q.push_back(next);
                    q[pointer_q++] = next;
                    pointer_q %= graph.n;
                }
            }
        }
    }
    return backward_counter;
}

static void reverse_local_update_linear(int t, const Graph &graph, double init_residual = 1) {
    bwd_idx.first.clean();
    bwd_idx.second.clean();

    static unordered_map<int, bool> idx;
    idx.clear();

    vector<int> q;
    q.reserve(graph.n);
    q.push_back(-1);
    unsigned long left = 1;

    double myeps = config.rmax;

    q.push_back(t);
    bwd_idx.second.insert(t, init_residual);

    idx[t] = true;
    while (left < q.size()) {
        int v = q[left];
        idx[v] = false;
        left++;
        if (bwd_idx.second[v] < myeps)
            break;

        if (bwd_idx.first.notexist(v))
            bwd_idx.first.insert(v, bwd_idx.second[v] * config.alpha);
        else
            bwd_idx.first[v] += bwd_idx.second[v] * config.alpha;

        double residual = (1 - config.alpha) * bwd_idx.second[v];
        bwd_idx.second[v] = 0;
        if (graph.gr[v].size() > 0) {
            for (int next : graph.gr[v]) {
                int cnt = graph.g[next].size();
                if (bwd_idx.second.notexist(next))
                    bwd_idx.second.insert(next, residual / cnt);
                else
                    bwd_idx.second[next] += residual / cnt;

                if (bwd_idx.second[next] > myeps && idx[next] != true) {
                    // put next into q if next is not in q
                    idx[next] = true;//(int) q.size();
                    q.push_back(next);
                }
            }
        }
    }
}

/*
 * ppr.clean();
    //先取出P
    int node_id;
    double reserve;
    for (long i = 0; i < fwd_idx.first.occur.m_num; i++) {
        node_id = fwd_idx.first.occur[i];
        reserve = fwd_idx.first[node_id];
        if (reserve > 0) {
            ppr.insert(node_id, reserve);
        }
    }
 */
int
reverse_local_update_linear_dht_topk(int s, const Graph &graph, double lowest_rmax, vector<int> &in_backward,
                                     vector<int> &in_next_backward, unordered_map<int, vector<int>> &backward_from) {
    Timer timer(BIPPR_QUERY);
    int backward_counter = 0;
    //////初始化！r
    if (backward_from[s].empty()) {
        multi_bwd_idx_p[s] = 0;
        multi_bwd_idx_r[s][s] = 1.0;
        backward_from[s].push_back(s);
    }
    vector<int> next_backward_from;
    next_backward_from.reserve(graph.n);

    for (auto &v: backward_from[s]) {
        in_backward[v] = s;
    }
    unsigned long i = 0;
    while (i < backward_from[s].size()) {
        int v = backward_from[s][i++];
        in_backward[v] = s;
        if (multi_bwd_idx_r[s][v] >= config2.rmax) {
            int out_neighbor = graph.gr[v].size();
            backward_counter += out_neighbor;
            if (s == v) {
                multi_bwd_idx_p[s] += multi_bwd_idx_r[s][v] * config.alpha;
            }
            double v_residue = (1 - config.alpha) * multi_bwd_idx_r[s][v];
            multi_bwd_idx_r[s].erase(v);//这里删除好还是等于0好？
            if (out_neighbor > 0) {
                for (int nei: graph.gr[v]) {
                    int cnt = graph.g[nei].size();
                    multi_bwd_idx_r[s][nei] += v_residue / cnt;
                    if (in_backward[nei] != s && multi_bwd_idx_r[s][nei] >= config2.rmax) {
                        backward_from[s].push_back(nei);
                        in_backward[nei] = s;
                    } else if (in_next_backward[nei] != s && multi_bwd_idx_r[s][nei] >= lowest_rmax) {
                        next_backward_from.push_back(nei);
                        in_next_backward[nei] = s;
                    }
                }
            }
        } else if (in_next_backward[v] != s && multi_bwd_idx_r[s][v] >= lowest_rmax) {
            next_backward_from.push_back(v);
            in_next_backward[v] = s;
        }
    }
    backward_from[s] = next_backward_from;
    return backward_counter;
#ifdef CHECK_PPR_VALUES
    display_ppr();
#endif
}


void init_bounds_self(const Graph &graph) {
    vector<int> out_neighbors(graph.n, -1);
    for (int j = 0; j < graph.n; ++j) {
        double p_back = 0;
        for (int out_nei:graph.g[j]) {
            out_neighbors[out_nei] = j;
            if (j == 281772) {
                INFO(out_nei);
            }
        }
        int re_num = 0;
        for (int in_nei:graph.gr[j]) {
            if (out_neighbors[in_nei] == j) {
                if (j == 281772) {
                    INFO(in_nei);
                }
                re_num++;
                p_back += 1.0 / graph.g[in_nei].size();
            }
        }

        p_back /= 1.0 * graph.g[j].size();
        p_back += (1 - re_num * 1.0 / graph.g[j].size());
        lower_bounds_self[j] = config.alpha + (1 - config.alpha) * (1 - config.alpha) * config.alpha * p_back;
        upper_bounds_self[j] = lower_bounds_self[j] +
                               (1 - config.alpha) * (1 - config.alpha) * (1 - config.alpha) * config.alpha *
                               (1 - p_back);
        if (j == 281772) {
            INFO(p_back, lower_bounds_self[j], upper_bounds_self[j]);
        }
    }
}

void forward_local_update_linear(int s, const Graph &graph, double &rsum, double rmax, double init_residual = 1.0) {
    fwd_idx.first.clean();//p
    fwd_idx.second.clean();//r

    static vector<bool> idx(graph.n);//标志是否在队列中
    std::fill(idx.begin(), idx.end(), false);

    double myeps = rmax;//config.rmax;

    vector<int> q;  //nodes that can still propagate forward
    q.reserve(graph.n);
    q.push_back(-1);
    unsigned long left = 1;
    q.push_back(s);

    // residual[s] = init_residual;
    fwd_idx.second.insert(s, init_residual);

    idx[s] = true;

    unsigned long long forward_counter = 0;
    while (left < (int) q.size()) {
        int v = q[left];
        idx[v] = false;
        left++;
        double v_residue = fwd_idx.second[v];
        fwd_idx.second[v] = 0;
        if (!fwd_idx.first.exist(v))//判断是否向量p中有点v，并进行更新
            fwd_idx.first.insert(v, v_residue * config.alpha);
        else
            fwd_idx.first[v] += v_residue * config.alpha;

        int out_neighbor = graph.g[v].size();
        rsum -= v_residue * config.alpha;
        if (out_neighbor == 0) {//如果邻居点没有出度，那么将这一点儿残差返回给源点！
            fwd_idx.second[s] += v_residue * (1 - config.alpha);
            if (graph.g[s].size() > 0 && fwd_idx.second[s] / graph.g[s].size() >= myeps && idx[s] != true) {
                idx[s] = true;
                q.push_back(s);
            }
            continue;
        }

        double avg_push_residual = ((1.0 - config.alpha) * v_residue) / out_neighbor;
        for (int next : graph.g[v]) {
            // total_push++;
            if (!fwd_idx.second.exist(next))
                fwd_idx.second.insert(next, avg_push_residual);
            else
                fwd_idx.second[next] += avg_push_residual;

            //if a node's' current residual is small, but next time it got a laerge residual, it can still be added into forward list
            //so this is correct
            if (fwd_idx.second[next] / graph.g[next].size() >= myeps && idx[next] != true) {
                idx[next] = true;//(int) q.size();
                q.push_back(next);
            }
        }
    }
}

void forward_local_update_linear_topk_dht(int s, const Graph &graph, double &rsum, double rmax, double lowest_rmax,
                                          vector<int> &forward_from) {
    double myeps = rmax;

    static vector<bool> in_forward(graph.n);
    static vector<bool> in_next_forward(graph.n);

    std::fill(in_forward.begin(), in_forward.end(), false);
    std::fill(in_next_forward.begin(), in_next_forward.end(), false);

    vector<int> next_forward_from;
    next_forward_from.reserve(graph.n);
    for (auto &v: forward_from)
        in_forward[v] = true;
    unsigned long long forward_counter = 0;
    unsigned long i = 0;
    while (i < forward_from.size()) {
        int v = forward_from[i];
        i++;
        in_forward[v] = false;
        if (fwd_idx.second[v] >= myeps) {
            int out_neighbor = graph.g[v].size();
            forward_counter += out_neighbor;
            double v_residue = fwd_idx.second[v];
            fwd_idx.second[v] = 0;
            if (!fwd_idx.first.exist(v)) {
                fwd_idx.first.insert(v, v_residue * config.alpha);
            } else {
                fwd_idx.first[v] += v_residue * config.alpha;
            }

            rsum -= v_residue * config.alpha;
            if (out_neighbor == 0) {
                fwd_idx.second[s] += v_residue * (1 - config.alpha);
                if (graph.g[s].size() > 0 && in_forward[s] != true && fwd_idx.second[s] >= myeps) {
                    forward_from.push_back(s);
                    in_forward[s] = true;
                } else {
                    if (graph.g[s].size() >= 0 && in_next_forward[s] != true &&
                        fwd_idx.second[s] >= lowest_rmax) {
                        next_forward_from.push_back(s);
                        in_next_forward[s] = true;
                    }
                }
                continue;
            }
            double avg_push_residual = ((1 - config.alpha) * v_residue) / out_neighbor;
            for (int next: graph.g[v]) {
                if (!fwd_idx.second.exist(next))
                    fwd_idx.second.insert(next, avg_push_residual);
                else
                    fwd_idx.second[next] += avg_push_residual;

                if (in_forward[next] != true && fwd_idx.second[next] >= myeps) {
                    forward_from.push_back(next);
                    in_forward[next] = true;
                } else {
                    if (in_next_forward[next] != true && fwd_idx.second[next] >= lowest_rmax) {
                        next_forward_from.push_back(next);
                        in_next_forward[next] = true;
                    }
                }
            }
        } else {
            if (in_next_forward[v] != true && fwd_idx.second[v] >= lowest_rmax) {
                next_forward_from.push_back(v);
                in_next_forward[v] = true;
            }
        }
    }
    INFO(forward_counter);
    num_total_fo += forward_counter;

    cout << "ratio of fo and ra:\t" << forward_counter / rsum / config.omega << endl;
    cout << "ratio of fo and esti fo:\t" << forward_counter / (1 / config.rmax) << endl;
    cout << "ratio of esti fo and ra:\t" << (1 / config.rmax) / (rsum * config.omega) << endl;
    cout << "ratio of esti fo and max ra:\t" << (1 / config.rmax) / (graph.n * config.rmax * config.omega) << endl;
    cout << "ratio of rsum and max:\t" << rsum / graph.n / config.rmax << endl;
    cout << "ratio of esti fo and esti bw:\t" << (1 / config.rmax) / (graph.n / config2.rmax) << endl;

    forward_from = next_forward_from;
}

int forward_local_update_linear_topk_dht2(int s, const Graph &graph, double &rsum, double rmax, double lowest_rmax,
                                          vector<int> &forward_from) {
    double myeps = rmax;

    static vector<bool> in_forward(graph.n);
    static vector<bool> in_next_forward(graph.n);

    std::fill(in_forward.begin(), in_forward.end(), false);
    std::fill(in_next_forward.begin(), in_next_forward.end(), false);

    vector<int> next_forward_from;
    next_forward_from.reserve(graph.n);
    for (auto &v: forward_from)
        in_forward[v] = true;
    unsigned long long forward_counter = 0;
    unsigned long i = 0;
    while (i < forward_from.size()) {
        int v = forward_from[i];
        i++;
        in_forward[v] = false;
        if (fwd_idx.second[v] / graph.g[v].size() >= myeps) {
            int out_neighbor = graph.g[v].size();
            forward_counter += out_neighbor;
            double v_residue = fwd_idx.second[v];
            fwd_idx.second[v] = 0;
            if (!fwd_idx.first.exist(v)) {
                fwd_idx.first.insert(v, v_residue * config.alpha);
            } else {
                fwd_idx.first[v] += v_residue * config.alpha;
            }

            rsum -= v_residue * config.alpha;
            if (out_neighbor == 0) {
                fwd_idx.second[s] += v_residue * (1 - config.alpha);
                if (graph.g[s].size() > 0 && in_forward[s] != true && fwd_idx.second[s] / graph.g[s].size() >= myeps) {
                    forward_from.push_back(s);
                    in_forward[s] = true;
                } else {
                    if (graph.g[s].size() >= 0 && in_next_forward[s] != true &&
                        fwd_idx.second[s] / graph.g[s].size() >= lowest_rmax) {
                        next_forward_from.push_back(s);
                        in_next_forward[s] = true;
                    }
                }
                continue;
            }
            double avg_push_residual = ((1 - config.alpha) * v_residue) / out_neighbor;
            for (int next: graph.g[v]) {
                if (!fwd_idx.second.exist(next))
                    fwd_idx.second.insert(next, avg_push_residual);
                else
                    fwd_idx.second[next] += avg_push_residual;

                if (in_forward[next] != true && fwd_idx.second[next] / graph.g[next].size() >= myeps) {
                    forward_from.push_back(next);
                    in_forward[next] = true;
                } else {
                    if (in_next_forward[next] != true && fwd_idx.second[next] / graph.g[next].size() >= lowest_rmax) {
                        next_forward_from.push_back(next);
                        in_next_forward[next] = true;
                    }
                }
            }
        } else {
            if (in_next_forward[v] != true && fwd_idx.second[v] / graph.g[v].size() >= lowest_rmax) {
                next_forward_from.push_back(v);
                in_next_forward[v] = true;
            }
        }
    }
    INFO(forward_counter);
    num_total_fo += forward_counter;

    forward_from = next_forward_from;
    return forward_counter;
}

void forward_local_update_linear_topk(int s, const Graph &graph, double &rsum, double rmax, double lowest_rmax,
                                      vector<int> &forward_from) {
    double myeps = rmax;

    static vector<bool> in_forward(graph.n);
    static vector<bool> in_next_forward(graph.n);

    std::fill(in_forward.begin(), in_forward.end(), false);
    std::fill(in_next_forward.begin(), in_next_forward.end(), false);

    vector<int> next_forward_from;
    next_forward_from.reserve(graph.n);
    for (auto &v: forward_from)
        in_forward[v] = true;

    unsigned long i = 0;
    while (i < forward_from.size()) {
        int v = forward_from[i];
        i++;
        in_forward[v] = false;
        if (fwd_idx.second[v] / graph.g[v].size() >= myeps) {
            int out_neighbor = graph.g[v].size();
            double v_residue = fwd_idx.second[v];
            fwd_idx.second[v] = 0;
            if (!fwd_idx.first.exist(v)) {
                fwd_idx.first.insert(v, v_residue * config.alpha);
            } else {
                fwd_idx.first[v] += v_residue * config.alpha;
            }

            rsum -= v_residue * config.alpha;
            if (out_neighbor == 0) {
                fwd_idx.second[s] += v_residue * (1 - config.alpha);
                if (graph.g[s].size() > 0 && in_forward[s] != true && fwd_idx.second[s] / graph.g[s].size() >= myeps) {
                    forward_from.push_back(s);
                    in_forward[s] = true;
                } else {
                    if (graph.g[s].size() >= 0 && in_next_forward[s] != true &&
                        fwd_idx.second[s] / graph.g[s].size() >= lowest_rmax) {
                        next_forward_from.push_back(s);
                        in_next_forward[s] = true;
                    }
                }
                continue;
            }
            double avg_push_residual = ((1 - config.alpha) * v_residue) / out_neighbor;
            for (int next: graph.g[v]) {
                if (!fwd_idx.second.exist(next))
                    fwd_idx.second.insert(next, avg_push_residual);
                else
                    fwd_idx.second[next] += avg_push_residual;

                if (in_forward[next] != true && fwd_idx.second[next] / graph.g[next].size() >= myeps) {
                    forward_from.push_back(next);
                    in_forward[next] = true;
                } else {
                    if (in_next_forward[next] != true && fwd_idx.second[next] / graph.g[next].size() >= lowest_rmax) {
                        next_forward_from.push_back(next);
                        in_next_forward[next] = true;
                    }
                }
            }
        } else {
            if (in_next_forward[v] != true && fwd_idx.second[v] / graph.g[v].size() >= lowest_rmax) {
                next_forward_from.push_back(v);
                in_next_forward[v] = true;
            }
        }
    }

    forward_from = next_forward_from;
}

extern double threshold;


bool if_stop() {
    // Timer tm(SORT_MAP);

    if (kth_ppr() >= 2.0 * config.delta)
        return true;

    if (config.delta >= threshold) return false;

    const static double error = 1.0 + config.epsilon;
    const static double error_2 = 1.0 + config.epsilon;

    topk_pprs.clear();
    topk_pprs.resize(config.k);
    topk_filter.clean();

    static vector<pair<int, double> > temp_bounds;
    temp_bounds.clear();
    temp_bounds.resize(lower_bounds.occur.m_num);
    int nodeid;
    for (int i = 0; i < lower_bounds.occur.m_num; i++) {
        nodeid = lower_bounds.occur[i];
        temp_bounds[i] = MP(nodeid, lower_bounds[nodeid]);
    }

    //sort topk nodes by lower bound
    partial_sort_copy(temp_bounds.begin(), temp_bounds.end(), topk_pprs.begin(), topk_pprs.end(),
                      [](pair<int, double> const &l, pair<int, double> const &r) { return l.second > r.second; });

    //for topk nodes, upper-bound/low-bound <= 1+epsilon
    double ratio = 0.0;
    double largest_ratio = 0.0;
    for (auto &node: topk_pprs) {
        topk_filter.insert(node.first, 1);
        ratio = upper_bounds[node.first] / lower_bounds[node.first];
        if (ratio > largest_ratio)
            largest_ratio = ratio;
        if (ratio > error_2) {
            return false;
        }
    }

    // INFO("ratio checking passed--------------------------------------------------------------");
    //for remaining NO. k+1 to NO. n nodes, low-bound of k > the max upper-bound of remaining nodes
    /*int actual_exist_ppr_num = lower_bounds.occur.m_num;
    if(actual_exist_ppr_num == 0) return true;
    int actual_k = min(actual_exist_ppr_num-1, (int) config.k-1);
    double low_bound_k = topk_pprs[actual_k].second;*/
    double low_bound_k = topk_pprs[config.k - 1].second;
    if (low_bound_k <= config.delta) {
        return false;
    }
    for (int i = 0; i < upper_bounds.occur.m_num; i++) {
        nodeid = upper_bounds.occur[i];
        if (topk_filter.exist(nodeid) || ppr[nodeid] <= 0)
            continue;

        double upper_temp = upper_bounds[nodeid];
        double lower_temp = lower_bounds[nodeid];
        if (upper_temp > low_bound_k * error) {
            if (upper_temp > (1 + config.epsilon) / (1 - config.epsilon) * lower_temp)
                continue;
            else {
                return false;
            }
        } else {
            continue;
        }
    }

    return true;
}

/*
 *
    do {
        num_iter_topk++;
        if (fora_flag) {
            fora_setting(graph.n, graph.m);

            {
                Timer timer(FWD_LU);
                forward_local_update_linear_topk(v, graph, rsum, config.rmax, lowest_delta_rmax,
                                                 forward_from); //forward propagation, obtain reserve and residual
            }

            compute_ppr_with_fwdidx_topk_with_bound(graph, rsum);
            if (first) {
                for (int j = 0; j < ppr.occur.m_num; ++j) {
                    dht.insert(ppr.occur[j], ppr[ppr.occur[j]]);
                }
            }
        }
        if (first || fora_flag == false) {//bippr
            //设置r_max和omega
            double old_omega = config2.omega;
            bippr_setting_lkx(graph.n, graph.m);
            if (config2.rmax < 1.0) {
                bippr_query_self(graph, lowest_delta_rmax_2, backward_from);
            }
            compute_ppr_with_bwdidx_with_bound(graph, old_omega, threshold);
        }
        first = false;
        int candidate_num = 0;
        stop = if_stop2(fora_flag, candidate_num);
        //选择算法
        bool f = true, b = true;
        if (kth_ppr() >= 2.0 * config.delta || config.delta <= min_delta) {
            f = false;
        }
        if (config2.epsilon <= min_epsilon) {
            b = false;
        }
        if (f == true && b == true) {
            double delta_new = max(min_delta, config.delta / 2.0);
            double rmax_new =
                    config.epsilon * sqrt(delta_new / 3 / graph.m / log(2 / config.pfail)) * config.rmax_scale;
            assert(rsum != 0);
            double omega =
                    rsum * (2 + config.epsilon) * log(2 / config.pfail) / delta_new / config.epsilon / config.epsilon;
            double cost_fora = omega + 1 / rmax_new - 1 / config.rmax;
            INFO(delta_new);
            INFO(cost_fora);
            double epsilon_new = max(min_epsilon, config2.epsilon / 2);
            rmax_new = epsilon_new * sqrt(graph.m * 1.0 * config2.delta / 3.0 / graph.n / log(2.0 / config2.pfail)) *
                       config2.rmax_scale;
            omega = rmax_new * 3 * log(2.0 / config2.pfail) / config2.delta / epsilon_new / epsilon_new;
            double cost_bippr = omega + 1 / rmax_new - 1 / config2.rmax;
            INFO(epsilon_new);
            INFO(candidate_num);
            INFO(candidate_num * cost_bippr);
            fora_flag = cost_fora < candidate_num * cost_bippr ? true : false;
        } else if (f == false && b == false) {
            break;
        } else {
            fora_flag = f;
        }
        if (fora_flag) {
            config.delta = max(min_delta, config.delta / 2);
        } else {
            config2.epsilon = max(min_epsilon, config2.epsilon / 2);
        }
    } while (!stop);

 */

bool if_stop_dht(const unordered_map<int, bool> &candidate, double raw_epsilon) {
    //思考这两个条件
    if (kth_ppr() >= 2.0 * config.delta / config.alpha) return true;
    if (config.delta >= threshold) return false;

    double true_epsilon = raw_epsilon;//(1 + config.epsilon) / (1 - config2.epsilon) - 1;
    double error = 1.0 + true_epsilon;

    topk_dhts.clear();
    topk_dhts.resize(config.k);
    topk_filter.clean();

    static vector<pair<int, double> > temp_bounds;
    temp_bounds.clear();
    temp_bounds.resize(candidate.size());
    int nodeid, cur = 0;
    for (auto item:candidate) {
        nodeid = item.first;
        temp_bounds[cur++] = MP(nodeid, lower_bounds_dht[nodeid]);
    }
    //sort topk nodes by lower bound
    partial_sort_copy(temp_bounds.begin(), temp_bounds.end(), topk_dhts.begin(), topk_dhts.end(),
                      [](pair<int, double> const &l, pair<int, double> const &r) { return l.second > r.second; });
    //display_topk_dht();
    //for topk nodes, upper-bound/low-bound <= 1+epsilon
    double ratio = 0.0;
    double largest_ratio = 0.0;
    for (auto &node: topk_dhts) {
        topk_filter.insert(node.first, 1);
        ratio = upper_bounds_dht[node.first] / lower_bounds_dht[node.first];
        if (largest_ratio < ratio) {
            largest_ratio = ratio;
        }
        if (ratio > error) {
            return false;
        }
    }
    INFO(largest_ratio);
    // INFO("ratio checking passed--------------------------------------------------------------");
    //for remaining NO. k+1 to NO. n nodes, low-bound of k > the max upper-bound of remaining nodes
    /*int actual_exist_ppr_num = lower_bounds.occur.m_num;
    if(actual_exist_ppr_num == 0) return true;
    int actual_k = min(actual_exist_ppr_num-1, (int) config.k-1);
    double low_bound_k = topk_pprs[actual_k].second;*/
    double low_bound_k = topk_dhts[config.k - 1].second;
    if (low_bound_k <= config.delta) {
        return false;
    }
    for (auto item:candidate) {
        nodeid = item.first;
        if (topk_filter.exist(nodeid) || dht[nodeid] <= 0)
            continue;

        double upper_temp = upper_bounds_dht[nodeid];
        double lower_temp = lower_bounds_dht[nodeid];
        if (upper_temp > low_bound_k * error) {
            if (upper_temp > (1 + true_epsilon) / (1 - true_epsilon) * lower_temp)
                continue;
            else {
                return false;
            }
        } else {
            continue;
        }
    }

    return true;

}

/*
 *     Timer timer(0);
    display_setting();
    const static double min_delta = 1.0 / graph.n;
    const static double init_delta = 1.0 / 4;

    threshold = (1.0 - config.ppr_decay_alpha) / pow(500, config.ppr_decay_alpha) /
                pow(graph.n, 1 - config.ppr_decay_alpha);//?
    const static double new_pfail = 1.0 / graph.n / graph.n / log(graph.n);

    config.pfail = new_pfail;  // log(1/pfail) -> log(1*n/pfail)
    config.pfail /= 2;
    config.epsilon = config.epsilon / (2 + config.epsilon);//(1+new_epsilon)/(1-new_epsilon)<=1+epsilon
    config2 = config;
    config.delta = init_delta;
    const static double lowest_delta_rmax =
            config.epsilon * sqrt(min_delta / 3 / graph.n / log(2 / new_pfail));//更新后的r_max
    const static double lowest_delta_rmax_2 =
            config.epsilon * sqrt(config2.delta / 3 / log(2.0 / new_pfail));

    double rsum = 1.0;
    //迭代起始点
    static vector<int> forward_from(graph.n);
    forward_from.push_back(v);
    static unordered_map<int, vector<int>> backward_from;
    //p和r等
    fwd_idx.first.clean();  //reserve
    fwd_idx.second.clean();  //residual
    fwd_idx.second.insert(v, rsum);
    multi_bwd_idx_p.clear();
    multi_bwd_idx_r.clear();
    ///config2的delta在后面确定！
    ///zero_ppr_upper_bound = 1.0;
    //先过一遍拿到上下限
    upper_bounds.reset_one_values();
    lower_bounds.reset_zero_values();
    init_bounds_self(graph);
 */
bool if_stop2(bool &fora_flag, int &candidate_num) {
    // Timer tm(SORT_MAP);
    bool stop = true;
    for (int j = 0; j < dht.occur.m_num; ++j) {
        int nodeid = dht.occur[j];
        if (dht.notexist(nodeid) || ppr.notexist(nodeid) || ppr_bi.notexist(nodeid)) continue;
        dht.insert(nodeid, ppr[nodeid] / ppr_bi[nodeid]);
        upper_bounds_dht.insert(nodeid, upper_bounds[nodeid] / lower_bounds_self[nodeid]);
        lower_bounds_dht.insert(nodeid, lower_bounds[nodeid] / upper_bounds_self[nodeid]);
        //cout<<nodeid<<":\t"<<upper_bounds[nodeid]<<"\t"<<lower_bounds[nodeid]<<"\t"<<upper_bounds_self[nodeid]<<"\t"<<lower_bounds_self[nodeid]<<"\t\t"<<upper_bounds_dht[nodeid]<<"\t"<<lower_bounds_dht[nodeid]<<endl;
    }

    //if (config.delta >= threshold) return false;

    const static double real_epsilon = (1.0 + config.epsilon) / (1.0 + config2.epsilon) - 1;

    topk_dhts.clear();
    topk_dhts.resize(config.k);
    topk_filter.clean();

    static vector<pair<int, double> > temp_bounds;
    temp_bounds.clear();
    temp_bounds.resize(dht.cur);
    int nodeid, cur = 0;
    for (int k = 0; k < dht.occur.m_num; ++k) {
        nodeid = dht.occur[k];
        if (dht.exist(nodeid))
            temp_bounds[cur++] = MP(nodeid, lower_bounds_dht[nodeid]);
    }
    //sort topk nodes by lower bound
    partial_sort_copy(temp_bounds.begin(), temp_bounds.end(), topk_dhts.begin(), topk_dhts.end(),
                      [](pair<int, double> const &l, pair<int, double> const &r) { return l.second > r.second; });

    //for topk nodes, upper-bound/low-bound <= 1+epsilon
    double ratio = 0.0;
    double largest_ratio = 0.0;
    cout << "topk:\t";
    for (auto &node: topk_dhts) {
        cout << node.first << "\t" << node.second << endl;
        topk_filter.insert(node.first, 1);
        ratio = upper_bounds_dht[node.first] / lower_bounds_dht[node.first];
        if (stop && ratio > largest_ratio)
            largest_ratio = ratio;
        if (ratio > real_epsilon + 1) {
            stop = false;
            //break;
            //return false;
        }
    }
    cout << endl;
    // INFO("ratio checking passed--------------------------------------------------------------");
    //for remaining NO. k+1 to NO. n nodes, low-bound of k > the max upper-bound of remaining nodes
    /*int actual_exist_ppr_num = lower_bounds.occur.m_num;
    if(actual_exist_ppr_num == 0) return true;
    int actual_k = min(actual_exist_ppr_num-1, (int) config.k-1);
    double low_bound_k = topk_pprs[actual_k].second;*/
    double low_bound_k = topk_dhts[config.k - 1].second;
    if (low_bound_k <= config.delta) {
        stop = false;
        //return false;
    }
    //确定有多少个bippr的候选点
    candidate_num = 0;
    INFO(low_bound_k);
    for (int m = 0; m < dht.occur.m_num; ++m) {
        nodeid = dht.occur[m];
        if (dht.notexist(nodeid))continue;
        candidate_num++;
        if (topk_filter.exist(nodeid)) continue;
        //cout<<nodeid<<"\t"<<dht[nodeid]<<"\t"<<upper_bounds_dht[nodeid]<<"\t"<<lower_bounds_dht[nodeid]<<endl;
        double upper_temp = upper_bounds_dht[nodeid];
        if (stop == true && upper_bounds_dht[nodeid] > low_bound_k * (1 + real_epsilon)) {
            if (upper_bounds_dht[nodeid] > (1 + real_epsilon) / (1 - real_epsilon) * lower_bounds_dht[nodeid]) {
                //cout<<nodeid<<"\t"<<dht[nodeid]<<"\t"<<upper_bounds_dht[nodeid]<<"\t"<<lower_bounds_dht[nodeid]<<endl;
                dht.erase(nodeid);
                candidate_num--;
            } else {
                stop = false;
            }
        } else if (stop == false && upper_bounds_dht[nodeid] < low_bound_k) {
            //cout<<nodeid<<"\t"<<dht[nodeid]<<"\t"<<upper_bounds_dht[nodeid]<<"\t"<<lower_bounds_dht[nodeid]<<endl;
            dht.erase(nodeid);
            candidate_num--;
        }
    }
    //计算两个成本
    return stop;
}

/*
ppr_bi[nodeid]=0.2
ppr_bi[nodeid]=0.21971
upper_bounds_self[nodeid]=0.274638

ppr_bi[nodeid]=0.285333
ppr_bi[nodeid]=0.285999
upper_bounds_self[nodeid]=0.302515
 * 262860:	0.0238329	0.0211118	0.228543	0.2		0.119165	0.0923754
 * 212918:	0.00761607	0.00647206	0.228271	0.2		0.0380803	0.0283525
topk:	62505	89073	60210	93989	50785	192704	262860	212918	179645	158108	32104	104294	169612	118111	41539	4518	33132	14475	202952	16937
topk:	62505	89073	60210	93989	50785	192704	212918	179645	158108	32104	104294	169612	118111	41539	33132	4518	14475	202952	16937	134832
62505:	0.201981	0.201348	0.22712	0.213355		0.946691	0.886527
50785:	0.0483986	0.0465708	0.211455	0.2		0.241993	0.220239
60210:	0.110057	0.107662	0.213385	0.200452		0.549043	0.504546
 * 62505:	0.201742	0.20135	0.22712	0.213355		0.945571	0.886534
50785:	0.0480772	0.0469477	0.211455	0.2		0.240386	0.222022
60210:	0.109465	0.108023	0.213385	0.200452		0.546092	0.506235
 */

inline double calculate_lambda(double rsum, double pfail, double upper_bound, long total_rw_num) {
    return 1.0 / 3 * log(2 / pfail) * rsum / total_rw_num +
           sqrt(4.0 / 9.0 * log(2.0 / pfail) * log(2.0 / pfail) * rsum * rsum +
                8 * total_rw_num * log(2.0 / pfail) * rsum * upper_bound)
           / 2.0 / total_rw_num;
}

double zero_ppr_upper_bound = 1.0;
double threshold = 0.0;

void set_ppr_bounds(const Graph &graph, double rsum, long total_rw_num) {
    Timer tm(100);

    const static double min_ppr = 1.0 / graph.n;
    const static double sqrt_min_ppr = sqrt(1.0 / graph.n);


    double epsilon_v_div = sqrt(2.67 * rsum * log(2.0 / config.pfail) / total_rw_num);
    double default_epsilon_v = epsilon_v_div / sqrt_min_ppr;

    int nodeid;
    double ub_eps_a;
    double lb_eps_a;
    double ub_eps_v;
    double lb_eps_v;
    double up_bound;
    double low_bound;
    // INFO(total_rw_num);
    // INFO(zero_ppr_upper_bound);
    //INFO(rsum, 1.0/config.pfail, log(2/config.pfail), zero_ppr_upper_bound, total_rw_num);
    zero_ppr_upper_bound = calculate_lambda(rsum, config.pfail, zero_ppr_upper_bound, total_rw_num);
    for (long i = 0; i < ppr.occur.m_num; i++) {
        nodeid = ppr.occur[i];
        assert(ppr[nodeid] > 0);
        if (ppr[nodeid] <= 0)
            continue;
        double reserve = 0.0;
        if (fwd_idx.first.exist(nodeid))
            reserve = fwd_idx.first[nodeid];
        double epsilon_a = 1.0;
        if (upper_bounds.exist(nodeid)) {
            assert(upper_bounds[nodeid] > 0.0);
            if (upper_bounds[nodeid] > reserve)
                //epsilon_a = calculate_lambda( rsum, config.pfail, upper_bounds[nodeid] - reserve, total_rw_num);
                epsilon_a = calculate_lambda(rsum, config.pfail, upper_bounds[nodeid] - reserve, total_rw_num);
            else
                epsilon_a = calculate_lambda(rsum, config.pfail, 1 - reserve, total_rw_num);
        } else {
            /*if(zero_ppr_upper_bound > reserve)
                epsilon_a = calculate_lambda( rsum, config.pfail, zero_ppr_upper_bound-reserve, total_rw_num);
            else
                epsilon_a = calculate_lambda( rsum, config.pfail, 1.0-reserve, total_rw_num);*/
            epsilon_a = calculate_lambda(rsum, config.pfail, 1.0 - reserve, total_rw_num);
        }

        ub_eps_a = ppr[nodeid] + epsilon_a;
        lb_eps_a = ppr[nodeid] - epsilon_a;
        if (!(lb_eps_a > 0))
            lb_eps_a = 0;

        double epsilon_v = default_epsilon_v;
        if (fwd_idx.first.exist(nodeid) && fwd_idx.first[nodeid] > min_ppr) {
            if (lower_bounds.exist(nodeid))
                reserve = max(reserve, lower_bounds[nodeid]);
            epsilon_v = epsilon_v_div / sqrt(reserve);
        } else {
            if (lower_bounds[nodeid] > 0)
                epsilon_v = epsilon_v_div / sqrt(lower_bounds[nodeid]);
        }


        ub_eps_v = 1.0;
        lb_eps_v = 0.0;
        if (1.0 - epsilon_v > 0) {
            ub_eps_v = ppr[nodeid] / (1.0 - epsilon_v);
            lb_eps_v = ppr[nodeid] / (1.0 + epsilon_v);
        }

        up_bound = min(min(ub_eps_a, ub_eps_v), 1.0);
        low_bound = max(max(lb_eps_a, lb_eps_v), reserve);
        if (up_bound > 0) {
            if (!upper_bounds.exist(nodeid))
                upper_bounds.insert(nodeid, up_bound);
            else
                upper_bounds[nodeid] = up_bound;
        }

        if (low_bound >= 0) {
            if (!lower_bounds.exist(nodeid))
                lower_bounds.insert(nodeid, low_bound);
            else
                lower_bounds[nodeid] = low_bound;
        }
        if (low_bound > up_bound) {
            INFO(epsilon_v);
            INFO(epsilon_a, calculate_lambda(config2.rmax, config.pfail, upper_bounds_self[nodeid], config2.omega));
            INFO(ub_eps_a, ub_eps_v, 1.0);
            INFO(lb_eps_a, lb_eps_v, reserve);
            INFO(multi_bwd_idx_p[nodeid], ppr_bi[nodeid]);
            INFO(low_bound, up_bound);
            cout << nodeid << endl;
        }
    }
}

void set_ppr_self_bounds(const Graph &graph, const unordered_map<int, bool> &candidate) {
    Timer tm(100);

    const static double min_ppr = 1.0 / graph.n;
    const static double sqrt_min_ppr = sqrt(1.0 / graph.n);


    double epsilon_v_div = sqrt(
            (2 + config2.epsilon * 2 / 3) * config2.rmax * log(2.0 / config2.pfail) / config2.omega);
    double default_epsilon_v = epsilon_v_div / sqrt_min_ppr;

    int nodeid;
    double ub_eps_a;
    double lb_eps_a;
    double ub_eps_v;
    double lb_eps_v;
    double up_bound;
    double low_bound;
    // INFO(total_rw_num);
    // INFO(zero_ppr_upper_bound);
    //INFO(rsum, 1.0/config.pfail, log(2/config.pfail), zero_ppr_upper_bound, total_rw_num);
    zero_ppr_upper_bound = calculate_lambda(config2.rmax, config2.pfail, zero_ppr_upper_bound, config2.omega);
    double large_ratio = 0;
    for (auto item:candidate) {
        int nodeid = item.first;
        assert(ppr_bi[nodeid] > 0);
        if (ppr_bi[nodeid] <= 0) continue;
        double reserve = 0;
        if (multi_bwd_idx_p[nodeid] > 0)
            reserve = multi_bwd_idx_p[nodeid];
        double epsilon_a = 1.0;
        if (upper_bounds_self.exist(nodeid)) {
            if (upper_bounds_self[nodeid] > reserve)
                //epsilon_a = calculate_lambda( rsum, config.pfail, upper_bounds[nodeid] - reserve, total_rw_num);
                epsilon_a = calculate_lambda(config2.rmax, config.pfail, upper_bounds_self[nodeid], config2.omega);
            else
                epsilon_a = calculate_lambda(config2.rmax, config.pfail, 1, config2.omega);
        } else {
            epsilon_a = calculate_lambda(config2.rmax, config.pfail, 1, config2.omega);
        }
        //INFO(epsilon_a,calculate_lambda(config2.rmax, config.pfail, upper_bounds_self[nodeid], config2.omega));
        ub_eps_a = ppr_bi[nodeid] + epsilon_a;
        lb_eps_a = ppr_bi[nodeid] - epsilon_a;
        if (!(lb_eps_a > 0))
            lb_eps_a = 0;
        double epsilon_v = default_epsilon_v;
        if (lower_bounds_self.exist(nodeid))
            reserve = max(reserve, lower_bounds_self[nodeid]);
        epsilon_v = epsilon_v_div / sqrt(reserve);
        //INFO(lower_bounds_self.exist(nodeid),lower_bounds_self[nodeid],epsilon_v);

        ub_eps_v = 1.0;
        lb_eps_v = 0.0;
        if (1.0 - epsilon_v > 0) {
            ub_eps_v = ppr_bi[nodeid] / (1.0 - epsilon_v);
            lb_eps_v = ppr_bi[nodeid] / (1.0 + epsilon_v);
        }

        up_bound = min(min(ub_eps_a, ub_eps_v), 1.0);
        low_bound = max(max(lb_eps_a, lb_eps_v), reserve);
        double old_up = upper_bounds_self[nodeid];
        double old_low = lower_bounds_self[nodeid];
        if (up_bound > 0) {
            if (!upper_bounds_self.exist(nodeid))
                upper_bounds_self.insert(nodeid, up_bound);
            else
                upper_bounds_self[nodeid] = up_bound;
        }

        if (low_bound >= 0) {
            if (!lower_bounds_self.exist(nodeid))
                lower_bounds_self.insert(nodeid, low_bound);
            else
                lower_bounds_self[nodeid] = low_bound;
        }

        if (low_bound > up_bound) {
            INFO(old_up, old_low, epsilon_v);
            INFO(epsilon_a, calculate_lambda(config2.rmax, config.pfail, upper_bounds_self[nodeid], config2.omega));
            INFO(ub_eps_a, ub_eps_v, 1.0);
            INFO(lb_eps_a, lb_eps_v, reserve);
            INFO(multi_bwd_idx_p[nodeid], ppr_bi[nodeid]);
            INFO(low_bound, up_bound);
            cout << nodeid << endl;
        }
        if (up_bound / low_bound > large_ratio)
            large_ratio = up_bound / low_bound;
    }
    INFO(large_ratio);
}

void set_dht_bounds(unordered_map<int, bool> &candidate) {
    //先计算界限，然后找第k大的，然后更新
    //初始化
    static vector<double> temp_dht;
    temp_dht.clear();
    //cout<<"node:\tppr\t"<<"ub_ppr\t"<<"lb_ppr\tratio\t"<<"ppr_bi\tub_self\t"<<"lb_self\tratio\t\t"<<"up_dht\tlb_dht\tratio"<<endl;
    if (candidate.empty()) {
        temp_dht.resize(upper_bounds.occur.m_num);
        for (int j = 0; j < ppr.occur.m_num; ++j) {
            int node = ppr.occur[j];
            assert(ppr.exist(node));
            if (ppr[node] < 0) {
                cout << "!!!!!" << node << ppr[node] << endl;
            }
            //dht.insert(node, ppr[node] / ppr_bi[node]);
            upper_bounds_dht.insert(node, upper_bounds[node] / lower_bounds_self[node]);
            lower_bounds_dht.insert(node, lower_bounds[node] / upper_bounds_self[node]);
            temp_dht[j] = lower_bounds_dht[node];

            /*out<<node<<":\t"<<ppr[node]<<"\t"<<upper_bounds[node]<<"\t"<<lower_bounds[node]<<"\t"<<upper_bounds[node]/lower_bounds[node]/(1+config.epsilon)*(1-config.epsilon)<<"\t"
            <<ppr[node]<<"\t"<<upper_bounds_self[node]<<"\t"<<lower_bounds_self[node]<<"\t"<<lower_bounds_self[node]/lower_bounds_self[node]/(1+config2.epsilon)*(1-config2.epsilon) <<"\t\t"
            <<upper_bounds_dht[node]<<"\t"<<lower_bounds_dht[node]<<"\t"<<upper_bounds_dht[node]/lower_bounds_dht[node]<<endl;*/

        }
        nth_element(temp_dht.begin(), temp_dht.begin() + config.k - 1, temp_dht.end(), cmp);
        double kth_dht_lb = temp_dht[config.k - 1];
        for (int k = 0; k < ppr.occur.m_num; ++k) {
            int nodeid = ppr.occur[k];
            if (upper_bounds_dht[nodeid] >= kth_dht_lb) {
                candidate.insert(make_pair(nodeid, true));
            }
        }
    } else {
        temp_dht.resize(candidate.size());
        int cur = 0;
        for (auto item:candidate) {
            int node = item.first;
            assert(ppr.exist(node));
            //dht.insert(node, ppr[node] / ppr_bi[node]);
            assert(item.second);
            upper_bounds_dht.insert(node, upper_bounds[node] / lower_bounds_self[node]);
            lower_bounds_dht.insert(node, lower_bounds[node] / upper_bounds_self[node]);
            temp_dht[cur++] = lower_bounds_dht[node];
/*1632786 1632784 1632783 1632764 1632761 1632752 1632747
            cout<<node<<":\t"<<upper_bounds[node]<<"\t"<<lower_bounds[node]<<"\t"<<upper_bounds[node]/lower_bounds[node]/(1+config.epsilon)*(1-config.epsilon)<<"\t"
                <<upper_bounds_self[node]<<"\t"<<lower_bounds_self[node]<<"\t"<<lower_bounds_self[node]/lower_bounds_self[node]/(1+config2.epsilon)*(1-config2.epsilon) <<"\t\t"
                <<upper_bounds_dht[node]<<"\t"<<lower_bounds_dht[node]<<"\t"<<upper_bounds_dht[node]/lower_bounds_dht[node]<<endl;
                */
        }
        nth_element(temp_dht.begin(), temp_dht.begin() + config.k - 1, temp_dht.end(), cmp);
        double kth_dht_lb = temp_dht[config.k - 1];
        for (auto iter = candidate.begin(); iter != candidate.end();) {
            int node = iter->first;
            assert(upper_bounds_dht.exist(node));
            if (upper_bounds_dht[node] < kth_dht_lb) {
                iter = candidate.erase(iter);
                multi_bwd_idx_r.erase(node);
            } else {
                ++iter;
            }
        }
    }
}

void set_martingale_bound(double lambda, unsigned long long total_num_rw, int t, double reserve, double rmax,
                          double pfail_star, double min_ppr, double m_omega) {
    double upper;
    double lower;
    double ey = total_num_rw * max(min_ppr - reserve, map_lower_bounds[t].second - reserve);
    double epsilon;
    if (m_omega > 0 && ey > 0) {
        epsilon = (sqrt(pow(rmax * pfail_star / 3, 2) - 2 * ey * pfail_star) - rmax * pfail_star / 3) / ey;
        upper = m_omega + lambda;
        if (1 - epsilon > 0 && upper > m_omega / (1 - epsilon))
            upper = m_omega / (1 - epsilon);
        upper /= total_num_rw;
        upper += reserve;
        lower = reserve + max(m_omega / (1 + epsilon), m_omega - lambda) / total_num_rw;
    } else {
        upper = reserve + (m_omega + lambda) / total_num_rw;
        lower = reserve + (m_omega - lambda) / total_num_rw;
    }

    // INFO(m_omega, ey, epsilon, lower, reserve);

    if (upper > 0 && upper < upper_bounds[t])
        upper_bounds[t] = upper;
    if (lower < 1 && lower > map_lower_bounds[t].second)
        map_lower_bounds[t].second = lower;
}

int reverse_local_update_topk(int s, int t, map<int, double> &map_reserve, double rmax,
                              unordered_map<int, double> &map_residual, const Graph &graph) {
    static vector<bool> bwd_flag(graph.n);

    vector<int> bwd_list;
    bwd_list.reserve(graph.n);
    unsigned long ptr = 0;

    for (auto &item: map_residual) {
        if (item.second >= rmax) {
            bwd_list.push_back(item.first);
            bwd_flag[item.first] = true;
        }
    }

    int push_count = 0;
    int v;
    while (ptr < bwd_list.size()) {
        v = bwd_list[ptr];
        bwd_flag[v] = false;
        ptr++;

        map_reserve[v] += map_residual[v] * config.alpha;

        double resi = (1 - config.alpha) * map_residual[v];
        map_residual.erase(v);
        push_count += graph.gr[v].size();
        for (int next: graph.gr[v]) {
            long cnt = graph.g[next].size();
            map_residual[next] += resi / cnt;
            if (map_residual[next] >= rmax && bwd_flag[next] != true) {
                bwd_flag[next] = true;
                bwd_list.push_back(next);
            }
        }
    }
    return push_count;
}

#endif
