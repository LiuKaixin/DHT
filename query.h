//Contributors: Sibo Wang, Renchi Yang
#ifndef FORA_QUERY_H
#define FORA_QUERY_H

#include "algo.h"
#include "graph.h"
#include "heap.h"
#include "config.h"
#include "build.h"
#include "string"
//#define CHECK_PPR_VALUES 1
//#define PRINT_PPR_VALUES 1
// #define CHECK_TOP_K_PPR 1
#define PRINT_PRECISION_FOR_DIF_K 1
// std::mutex mtx;

void montecarlo_query(int v, const Graph &graph) {
    Timer timer(MC_QUERY);

    rw_counter.clean();
    ppr.reset_zero_values();

    {
        Timer tm(RONDOM_WALK);
        num_total_rw += config.omega;
        for (unsigned long i = 0; i < config.omega; i++) {
            int destination = random_walk(v, graph);
            if (!rw_counter.exist(destination))
                rw_counter.insert(destination, 1);
            else
                rw_counter[destination] += 1;
        }
    }

    int node_id;
    for (long i = 0; i < rw_counter.occur.m_num; i++) {
        node_id = rw_counter.occur[i];
        ppr[node_id] = rw_counter[node_id] * 1.0 / config.omega;
    }

#ifdef CHECK_PPR_VALUES
    display_ppr();
#endif
}

void montecarlo_query2(int v, const Graph &graph) {}

void montecarlo_query_topk(int v, const Graph &graph) {
    Timer timer(0);

    rw_counter.clean();
    ppr.clean();

    {
        Timer tm(RONDOM_WALK);
        num_total_rw += config.omega;
        for (unsigned long i = 0; i < config.omega; i++) {
            int destination = random_walk(v, graph);
            if (!rw_counter.exist(destination))
                rw_counter.insert(destination, 1);
            else
                rw_counter[destination] += 1;
        }
    }

    int node_id;
    for (long i = 0; i < rw_counter.occur.m_num; i++) {
        node_id = rw_counter.occur[i];
        if (rw_counter.occur[i] > 0)
            ppr.insert(node_id, rw_counter[node_id] * 1.0 / config.omega);
    }
}


void montecarlo_query_dht(int v, const Graph &graph) {
    Timer timer(MC_DHT_QUERY);

    rw_counter.reset_zero_values();
    dht.reset_zero_values();
    unordered_map<int, pair<int, int>> occur;

    {
        Timer tm(RONDOM_WALK);
        INFO(config.omega);
        num_total_rw += config.omega;
        for (unsigned long i = 0; i < config.omega; i++) {
            random_walk_dht(v, i, graph, occur);
        }
    }

    for (auto item:occur) {
        int node_id = item.first;
        dht[node_id] = item.second.second * 1.0 / config.omega;
    }

#ifdef CHECK_PPR_VALUES
    display_dht();
#endif
}

void montecarlo_query_dht_topk(int v, const Graph &graph) {
    Timer timer(0);

    rw_counter.reset_zero_values();
    unordered_map<int, pair<int, int>> occur;//为每个点建立索引，pair中第一个是上一次的路径号，下一个是出现过的次数

    {
        Timer tm(RONDOM_WALK);
        INFO(config.omega);
        num_total_rw += config.omega;
        for (unsigned long i = 0; i < config.omega; i++) {
            random_walk_dht(v, i, graph, occur);
        }
    }
    dht.clean();
    for (auto item:occur) {
        int node_id = item.first;
        //INFO(node_id,item.second.second);
        dht.insert(node_id, item.second.second * 1.0 / config.omega);
        //dht[node_id] = item.second.second * 1.0 / config.omega;
    }

#ifdef CHECK_PPR_VALUES
    display_dht();
#endif
}

void global_iteration_query(int v, const Graph &graph) {
    Timer timer(GI_QUERY);

    dht.reset_zero_values();
    vector<double> dht_tmp(graph.n, 0);
    for (int l = 0; l < dht.m_num; ++l) {
        double max_error = 0;
        bool first = true;
        int k = 0;
        dht_tmp.clear();
        //vector<int> dht_to_l;
        do {
            k++;
            global_iteration(v, dht_tmp, graph);
            if (first) {
                first = false;
                max_error = *max_element(dht_tmp.begin(), dht_tmp.end());
            }
        } while (pow(1 - config.alpha, k) / config.alpha * max_error > config.epsilon * config.delta);
        dht.insert(l, dht_tmp[l]);
    }
#ifdef CHECK_PPR_VALUES
    display_dht();
#endif
}

void dne_query(int v, const Graph &graph) {
    Timer timer(DNE_QUERY);

    dht.reset_zero_values();
    double deg_graph = graph.m / graph.n;
    int dne_m = 30000;//ceil(pow(deg_graph, log(graph.n) / log(deg_graph / (1 - config.alpha))));
    INFO(dne_m);
    dhe_query_basic(v, v, dne_m, graph);
    /*
    for (int l = 0; l < dht.m_num; ++l) {
        dht.insert(l, dhe_query_basic(v, l, dne_m, graph));
    }*/
#ifdef CHECK_PPR_VALUES
    display_dht();
#endif
}

void bippr_query(int v, const Graph &graph) {
    Timer timer(BIPPR_QUERY);

    rw_counter.clean();
    ppr.reset_zero_values();

    {
        Timer tm(RONDOM_WALK);
        num_total_rw += config.omega;
        INFO(config.omega);
        for (unsigned long i = 0; i < config.omega; i++) {
            int destination = random_walk(v, graph);
            if (!rw_counter.exist(destination))
                rw_counter.insert(destination, 1);
            else
                rw_counter[destination] += 1;
        }
    }

    INFO(config.rmax);
    if (config.rmax < 1.0) {
        Timer tm(BWD_LU);
        for (long i = 0; i < graph.n; i++) {
            reverse_local_update_linear(i, graph);
            // if(backresult.first[v] ==0 && backresult.second.size()==0){
            if ((!bwd_idx.first.exist(v) || 0 == bwd_idx.first[v]) && 0 == bwd_idx.second.occur.m_num) {
                continue;
            }
            ppr[i] += bwd_idx.first[v];
            // for(auto residue: backresult.second){
            for (long j = 0; j < bwd_idx.second.occur.m_num; j++) {
                // ppr[i]+=counts[residue.first]*1.0/config.omega*residue.second;
                int nodeid = bwd_idx.second.occur[j];
                double residual = bwd_idx.second[nodeid];
                int occur;
                if (!rw_counter.exist(nodeid))
                    occur = 0;
                else
                    occur = rw_counter[nodeid];

                ppr[i] += occur * 1.0 / config.omega * residual;
            }
        }
    } else {
        int node_id;
        for (long i = 0; i < rw_counter.occur.m_num; i++) {
            node_id = rw_counter.occur[i];
            ppr[node_id] = rw_counter[node_id] * 1.0 / config.omega;
        }
    }
#ifdef CHECK_PPR_VALUES
    display_ppr();
#endif
}

void bippr_query_fb_raw(const Graph &graph, double threshold) {
    ppr_bi.initialize(graph.n);
    //ppr_self
    rw_counter.initialize(graph.n);
    fill(bwd_idx.first.occur.m_data, bwd_idx.first.occur.m_data + graph.n, -1);
    fill(bwd_idx.second.occur.m_data, bwd_idx.second.occur.m_data + graph.n, -1);
    fill(rw_counter.occur.m_data, rw_counter.occur.m_data + graph.n, -1);
    vector<int> idx(graph.n, -1), node_with_r(graph.n), q(graph.n);
    int pointer_r = 0;
    int total_rw_b = 0;
    //static unordered_map<int, int > idx;///
    INFO(config2.omega);
    assert(config2.rmax < 1);
    unsigned long long backward_counter = 0, old_num_total_rw = num_total_rw;
    int counter = 0;
    for (int k = 0; k < ppr.m_num; ++k) {
        if (ppr[k] <= threshold)continue;
        counter++;
        {
            Timer tm(RONDOM_WALK);
            //double residual = fwd_idx.second[k] > 0 ? fwd_idx.second[k] : 0;
            assert(residual < config.rmax);
            assert(residual < config2.rmax);
            unsigned long num_s_rw = ceil(config2.omega);
            num_total_rw += num_s_rw;
            total_rw_b += num_s_rw;
            for (unsigned long i = 0; i < num_s_rw; i++) {
                int destination = random_walk(k, graph);
                if (rw_counter.occur[destination] != k) {
                    rw_counter.occur[destination] = k;
                    rw_counter[destination] = 1;
                } else {
                    ++rw_counter[destination];
                }
            }
        }
        if (config2.rmax < 1.0) {
            Timer tm(BWD_LU);
            backward_counter += reverse_local_update_linear_dht(k, graph, idx, node_with_r, pointer_r, q);
            assert(bwd_idx.first[k] > 0 && bwd_idx.first.occur[k] == k);
            ppr_bi.insert(k, bwd_idx.first[k]);
            for (int j = 0; j < pointer_r; ++j) {
                int nodeid = node_with_r[j];
                double residual = bwd_idx.second[nodeid];
                if (rw_counter.occur[nodeid] == k) {
                    ppr_bi[k] += rw_counter[nodeid] * 1.0 / config2.omega * residual;
                }
            }
        } else {
            assert(rw_counter.occur[k] == k);
            if (rw_counter.occur[k] == k) {
                ppr_bi.insert(k, rw_counter[k] / 1.0 / config2.omega);
            }
        }
    }
    dht.initialize(graph.n);
    for (int l = 0; l < ppr_bi.occur.m_num; ++l) {
        int nodeid = ppr_bi.occur[l];
        dht.insert(nodeid, ppr[nodeid] / ppr_bi[nodeid]);
    }
    INFO(counter);
    INFO(total_rw_b);
}

void bippr_query_fb_raw_topk_with_bound(unordered_map<int, bool> &candidate, const Graph &graph) {
    //对candidate中的节点计算bippr到自己的
    //ppr_self
    rw_counter.initialize(graph.n);
    fill(bwd_idx.first.occur.m_data, bwd_idx.first.occur.m_data + graph.n, -1);
    fill(bwd_idx.second.occur.m_data, bwd_idx.second.occur.m_data + graph.n, -1);
    fill(rw_counter.occur.m_data, rw_counter.occur.m_data + graph.n, -1);
    vector<int> idx(graph.n, -1), node_with_r(graph.n), q(graph.n);
    int pointer_r = 0;
    //static unordered_map<int, int > idx;///
    INFO(config2.omega);
    assert(config2.rmax < 1);
    unsigned long long backward_counter = 0, old_num_total_rw = num_total_rw;
    int counter = 0;
    for (auto item:candidate) {
        int node = item.first;
        counter++;
        {
            Timer tm(RONDOM_WALK);
            //double residual = fwd_idx.second[k] > 0 ? fwd_idx.second[k] : 0;
            unsigned long num_s_rw = ceil(config2.omega);
            num_total_rw += num_s_rw;
            for (unsigned long i = 0; i < num_s_rw; i++) {
                int destination = random_walk(node, graph);
                if (rw_counter.occur[destination] != node) {
                    rw_counter.occur[destination] = node;
                    rw_counter[destination] = 1;
                } else {
                    ++rw_counter[destination];
                }
            }
        }
        if (config2.rmax < 1.0) {
            Timer tm(BWD_LU);
            backward_counter += reverse_local_update_linear_dht(node, graph, idx, node_with_r, pointer_r, q);
            ppr_bi.insert(node, bwd_idx.first[node]);
            for (int j = 0; j < pointer_r; ++j) {
                int nodeid = node_with_r[j];
                double residual = bwd_idx.second[nodeid];
                if (rw_counter.occur[nodeid] == node) {
                    ppr_bi[node] += rw_counter[nodeid] * 1.0 / config2.omega * residual;
                }
            }
        } else {
            if (rw_counter.occur[node] == node) {
                ppr_bi.insert(node, rw_counter[node] / 1.0 / config2.omega);
            }
        }
    }
    set_ppr_self_bounds(graph, candidate);
}

void compute_ppr_with_reserve() {
    ppr.clean();
    int node_id;
    double reserve;
    for (long i = 0; i < fwd_idx.first.occur.m_num; i++) {
        node_id = fwd_idx.first.occur[i];
        reserve = fwd_idx.first[node_id];
        if (reserve)
            ppr.insert(node_id, reserve);
    }
}

void bippr_query_with_fora(const Graph &graph, double check_rsum) {
    ppr.reset_zero_values();
    //先取出P
    int node_id;
    double reserve;
    for (long i = 0; i < fwd_idx.first.occur.m_num; i++) {
        node_id = fwd_idx.first.occur[i];
        reserve = fwd_idx.first[node_id];
        ppr[node_id] = reserve;
    }
    if (check_rsum == 0.0)
        return;

    unsigned long long num_random_walk = config.omega * check_rsum;//这里的omega并不是真正的Omega，num_random_walk才是
    INFO(num_random_walk);

    //ppr_self
    ppr_bi.initialize(graph.n);
    rw_counter.initialize(graph.n);
    fill(bwd_idx.first.occur.m_data, bwd_idx.first.occur.m_data + graph.n, -1);
    fill(bwd_idx.second.occur.m_data, bwd_idx.second.occur.m_data + graph.n, -1);
    fill(rw_counter.occur.m_data, rw_counter.occur.m_data + graph.n, -1);
    vector<int> idx(graph.n, -1), node_with_r(graph.n), q(graph.n);
    int pointer_r = 0;
    //static unordered_map<int, int > idx;///
    INFO(config2.omega);
    assert(config2.rmax < 1);
    unsigned long long backward_counter = 0, old_num_total_rw = num_total_rw;
    int counter = 0;
    unsigned long num_s_rw_real;
    for (int k = 0; k < ppr.m_num; ++k) {
        //执行一遍，如果r>0或者ppr>delta则执行，最后检查一遍。
        if (ppr[k] <= config.delta && (fwd_idx.second.notexist(k) || fwd_idx.second[k] <= 0))continue;
        counter++;
        {
            Timer tm(RONDOM_WALK);
            double residual = fwd_idx.second[k] > 0 ? fwd_idx.second[k] : 0;
            unsigned long num_s_rw = ceil(residual / check_rsum * num_random_walk);
            if (ppr[k]>config.delta)
                num_s_rw_real = ceil(config2.omega) > num_s_rw ? ceil(config2.omega) : num_s_rw;
            else
                num_s_rw_real=num_s_rw;

            double ppr_incre = residual / num_s_rw_real;
            num_total_rw += num_s_rw_real;
            for (unsigned long i = 0; i < num_s_rw_real; i++) {
                int destination = random_walk(k, graph);
                ppr[destination] += ppr_incre;
                if (ppr[k]>config.delta){
                    if (rw_counter.occur[destination] != k) {
                        rw_counter.occur[destination] = k;
                        rw_counter[destination] = 1;
                    } else {
                        ++rw_counter[destination];
                    }
                }
            }
        }
        if (ppr[k]<=config.delta) continue;
        if (config2.rmax < 1.0) {
            Timer tm(BWD_LU);
            backward_counter += reverse_local_update_linear_dht(k, graph, idx, node_with_r, pointer_r, q);
            assert(bwd_idx.first[k] > 0 && bwd_idx.first.occur[k] == k);
            ppr_bi.insert(k, bwd_idx.first[k]);
            for (int j = 0; j < pointer_r; ++j) {
                int nodeid = node_with_r[j];
                double residual = bwd_idx.second[nodeid];
                if (rw_counter.occur[nodeid] == k) {
                    ppr_bi[k] += rw_counter[nodeid] * 1.0 / num_s_rw_real * residual;
                }
            }
        } else {
            assert(rw_counter.occur[k] == k);
            if (rw_counter.occur[k] == k) {
                ppr_bi.insert(k, rw_counter[k] / 1.0 / num_s_rw_real);
            }
        }
    }
    for (int m = 0; m < ppr.m_num; ++m) {
        if (ppr[m]>config.delta&&ppr_bi.notexist(m)){
            {
                Timer tm(RONDOM_WALK);
                num_s_rw_real= ceil(config2.omega);
                num_total_rw += num_s_rw_real;
                for (unsigned long i = 0; i < num_s_rw_real; i++) {
                    int destination = random_walk(m, graph);
                    if (rw_counter.occur[destination] != m) {
                        rw_counter.occur[destination] = m;
                        rw_counter[destination] = 1;
                    } else {
                        ++rw_counter[destination];
                    }
                }
            }
            if (config2.rmax < 1.0) {
                Timer tm(BWD_LU);
                backward_counter += reverse_local_update_linear_dht(m, graph, idx, node_with_r, pointer_r, q);
                ppr_bi.insert(m, bwd_idx.first[m]);
                for (int j = 0; j < pointer_r; ++j) {
                    int nodeid = node_with_r[j];
                    double residual = bwd_idx.second[nodeid];
                    if (rw_counter.occur[nodeid] == m) {
                        ppr_bi[m] += rw_counter[nodeid] * 1.0 / num_s_rw_real * residual;
                    }
                }
            } else {
                if (rw_counter.occur[m] == m) {
                    ppr_bi.insert(m, rw_counter[m] / 1.0 / num_s_rw_real);
                }
            }
        }
    }
    dht.initialize(graph.n);
    for (int l = 0; l < ppr_bi.occur.m_num; ++l) {
        int nodeid = ppr_bi.occur[l];
        dht.insert(nodeid, ppr[nodeid] / ppr_bi[nodeid]);
    }
    INFO(counter);
    INFO(num_total_rw - old_num_total_rw);
    INFO(backward_counter);
    cout << "ratio of bw and ra:\t" << backward_counter / fwd_idx.second.occur.m_num / config2.omega << endl;
    cout << "ratio of bw and true ra:\t" << backward_counter * 1.0 / (num_total_rw - old_num_total_rw) << endl;
    cout << "ratio of bw and esti bw:\t" << backward_counter / (graph.n / config2.rmax) << endl;
    cout << "ratio of ra and esti ra:\t" << (num_total_rw - old_num_total_rw) / graph.n / config2.omega << endl;
    cout << "ratio of esti bw and esti ra:\t" << (1 / config2.rmax) / config2.omega << endl;
    num_total_bi += backward_counter;
#ifdef CHECK_PPR_VALUES
    display_ppr();
#endif
}

int bippr_query_candidate(const Graph &graph, const unordered_map<int, bool> &candidate, const double &lowest_rmax,
                          unordered_map<int, vector<int>> &backward_from) {
    Timer timer(BIPPR_QUERY);
    ppr_bi.clean();
    static vector<int> in_backward(graph.n);
    static vector<int> in_next_backward(graph.n);
    std::fill(in_backward.begin(), in_backward.end(), -1);
    std::fill(in_next_backward.begin(), in_next_backward.end(), -1);
    fill(rw_counter.occur.m_data, rw_counter.occur.m_data + graph.n, -1);
    unsigned long long backward_counter = 0, old_num_total_rw = num_total_rw;
    for (auto item:candidate) {
        int node_id = item.first;
        {
            Timer tm(DFS_CYCLE);
            num_total_rw += ceil(config2.omega);
            for (unsigned long i = 0; i < ceil(config2.omega); i++) {
                int destination = random_walk(node_id, graph);
                if (rw_counter.occur[destination] != node_id) {
                    rw_counter.occur[destination] = node_id;
                    rw_counter[destination] = 1;
                } else {
                    ++rw_counter[destination];
                }
            }
        }
        if (config2.rmax < 1.0) {
            Timer tm(BWD_LU);
            backward_counter += reverse_local_update_linear_dht_topk(node_id, graph, lowest_rmax, in_backward,
                                                                     in_next_backward, backward_from);
            ppr_bi.insert(node_id, multi_bwd_idx_p[node_id]);
            for (auto item:multi_bwd_idx_r[node_id]) {
                int node = item.first;
                double residual = item.second;
                if (rw_counter.occur[node] == node_id) {
                    ppr_bi[node_id] += rw_counter[node] * 1.0 / config2.omega * residual;
                }
            }
        } else {
            if (rw_counter.occur[node_id] == node_id) {
                ppr_bi.insert(node_id, rw_counter[node_id] / 1.0 / config2.omega);
            }
        }
    }
    set_ppr_self_bounds(graph, candidate);
    /*
    INFO(num_total_rw - old_num_total_rw);
    INFO(backward_counter, candidate.size(), 1 / config2.rmax, config2.epsilon, config2.delta);

    cout << "ratio of bw and ra:\t" << backward_counter / candidate.size() / config2.omega << endl;
    cout << "ratio of bw and true ra:\t" << backward_counter * 1.0 / (num_total_rw - old_num_total_rw) << endl;
    cout << "ratio of ra and esti ra:\t" << (num_total_rw - old_num_total_rw) / candidate.size() / config2.omega
         << endl;*/
    num_total_bi += backward_counter;
    return backward_counter;
}
int bippr_query_candidate_with_idx(const Graph &graph, const unordered_map<int, bool> &candidate, const double &lowest_rmax,
                          unordered_map<int, vector<int>> &backward_from) {
    Timer timer(BIPPR_QUERY);
    ppr_bi.clean();
    static vector<int> in_backward(graph.n);
    static vector<int> in_next_backward(graph.n);
    std::fill(in_backward.begin(), in_backward.end(), -1);
    std::fill(in_next_backward.begin(), in_next_backward.end(), -1);
    fill(rw_counter.occur.m_data, rw_counter.occur.m_data + graph.n, -1);
    unsigned long long backward_counter = 0, old_num_total_rw = num_total_rw;
    for (auto item:candidate) {
        int node_id = item.first;
        if (config2.rmax < 1.0) {
            Timer tm(BWD_LU);
            backward_counter += reverse_local_update_linear_dht_topk(node_id, graph, lowest_rmax, in_backward,
                                                                     in_next_backward, backward_from);
            ppr_bi.insert(node_id, multi_bwd_idx_p[node_id]);
            {
                Timer tm(DFS_CYCLE);
                int num_s_rw=ceil(config2.omega);
                num_total_rw += num_s_rw;
                int index_size=rw_index[node_id].size();
                if (num_s_rw>index_size){//如果不够，就先用索引中然后在线生成
                    for (int j = 0; j < index_size; ++j) {
                        int des =rw_index[node_id][j];
                        if (multi_bwd_idx_r[node_id].find(des)!=multi_bwd_idx_r[node_id].end()){
                            ppr_bi[node_id] += multi_bwd_idx_r[node_id][des]/num_s_rw;
                        }
                    }
                    for (int k = 0; k < num_s_rw - index_size; ++k) {//在线生成同时更新索引
                        int des = random_walk(node_id, graph);
                        rw_index[node_id].emplace_back(des);
                        if (multi_bwd_idx_r[node_id].find(des)!=multi_bwd_idx_r[node_id].end()){
                            ppr_bi[node_id] += multi_bwd_idx_r[node_id][des]/num_s_rw;
                        }
                    }
                }else{//如果够了就只用索引
                    for (int j = 0; j < num_s_rw; ++j) {
                        int des = rw_index[node_id][j];
                        if (multi_bwd_idx_r[node_id].find(des)!=multi_bwd_idx_r[node_id].end()){
                            ppr_bi[node_id] += multi_bwd_idx_r[node_id][des]/num_s_rw;
                        }
                    }
                }
            }
        } else {
            {
                Timer tm(DFS_CYCLE);
                int num_s_rw=ceil(config2.omega);
                num_total_rw += num_s_rw;
                int index_size=rw_index[node_id].size();
                ppr_bi.insert(node_id, 0);
                INFO(node_id,ppr_bi[node_id]);
                if (num_s_rw>index_size){//如果不够，就先用索引中然后在线生成
                    for (int j = 0; j < index_size; ++j) {
                        int des =rw_index[node_id][j];
                        if (des==node_id){
                            ppr_bi[node_id] += 1.0/num_s_rw;
                        }
                    }
                    for (int k = 0; k < num_s_rw - index_size; ++k) {//在线生成同时更新索引
                        int des = random_walk(node_id, graph);
                        rw_index[node_id].emplace_back(des);
                        if (des==node_id){
                            ppr_bi[node_id] += 1.0/num_s_rw;
                        }
                    }
                }else{//如果够了就只用索引
                    for (int j = 0; j < num_s_rw; ++j) {
                        int des = rw_index[node_id][j];
                        if (des==node_id){
                            ppr_bi[node_id] += 1.0/num_s_rw;
                        }
                    }
                }
                INFO(node_id,ppr_bi[node_id]);
            }
        }
    }
    set_ppr_self_bounds(graph, candidate);
    /*
    INFO(num_total_rw - old_num_total_rw);
    INFO(backward_counter, candidate.size(), 1 / config2.rmax, config2.epsilon, config2.delta);

    cout << "ratio of bw and ra:\t" << backward_counter / candidate.size() / config2.omega << endl;
    cout << "ratio of bw and true ra:\t" << backward_counter * 1.0 / (num_total_rw - old_num_total_rw) << endl;
    cout << "ratio of ra and esti ra:\t" << (num_total_rw - old_num_total_rw) / candidate.size() / config2.omega
         << endl;*/
    num_total_bi += backward_counter;
    return backward_counter;
}

bool bippr_query_with_fora_topk(const Graph &graph, double check_rsum, unordered_map<int, bool> &candidate,
                                double lowest_rmax, unordered_map<int, vector<int>> &backward_from) {
    compute_ppr_with_reserve();
    for (auto iter = candidate.begin(); iter != candidate.end();) {
        int node = iter->first;
        if (ppr.notexist(node)) {
            iter = candidate.erase(iter);
            multi_bwd_idx_r.erase(node);
        } else {
            ++iter;
        }
    }
    if (check_rsum == 0.0)
        return false;

    unsigned long long num_random_walk = config.omega * check_rsum;//这里的omega并不是真正的Omega，num_random_walk才是
    INFO(num_random_walk);
    static bool run_bippr = false;
    if (!run_bippr) {
        run_bippr = compare_fora_bippr_cost(candidate.size(), graph.m, check_rsum);
    }
    //ppr_self
    static vector<int> in_backward(graph.n);
    static vector<int> in_next_backward(graph.n);
    std::fill(in_backward.begin(), in_backward.end(), -1);
    std::fill(in_next_backward.begin(), in_next_backward.end(), -1);
    fill(rw_counter.occur.m_data, rw_counter.occur.m_data + graph.n, -1);
    unsigned long long backward_counter = 0, old_num_total_rw = num_total_rw;
    long long real_total_rw = 0;//计算界限需要用
    int counter = 0, node_id;
    //带r的需要抽样随机游走路径，有PPR的需要并且run_bippr为真需要抽样随机游走路径
    for (int j = 0; j < graph.n; ++j) {
        if (candidate.find(j) != candidate.end() && ppr[j] < 0) {
            INFO(j, ppr[j], candidate.at(j));
        }
        if (ppr.notexist(j) && (fwd_idx.second.notexist(j) || fwd_idx.second[j] <= 0))continue;
        node_id = j;
        unsigned long num_s_rw_real;
        if ((!run_bippr || candidate.find(node_id) == candidate.end()) && fwd_idx.second[node_id] <= 0) continue;
        counter++;
        {
            Timer tm(RONDOM_WALK);
            double residual = fwd_idx.second[node_id] > 0 ? fwd_idx.second[node_id] : 0;
            unsigned long num_s_rw = ceil(residual / check_rsum * num_random_walk);
            real_total_rw += num_s_rw;
            num_s_rw_real = num_s_rw;
            if (run_bippr && candidate.find(node_id) != candidate.end() && ceil(config2.omega) > num_s_rw) {
                num_s_rw_real = ceil(config2.omega);
            }
            double a_s = residual / check_rsum * num_random_walk / num_s_rw_real;

            double ppr_incre = a_s * check_rsum / num_random_walk;// * num_s_rw / num_s_rw_real;
            num_total_rw += num_s_rw_real;
            for (unsigned long i = 0; i < num_s_rw_real; i++) {
                int destination = random_walk(node_id, graph);
                if (residual > 0) {
                    if (ppr.notexist(destination)) {
                        ppr.insert(destination, ppr_incre);
                    } else {
                        ppr[destination] += ppr_incre;
                    }
                }

                if (rw_counter.occur[destination] != node_id) {
                    rw_counter.occur[destination] = node_id;
                    rw_counter[destination] = 1;
                } else {
                    ++rw_counter[destination];
                }
            }
        }
        if (!run_bippr || candidate.find(node_id) == candidate.end())continue;
        if (config2.rmax < 1.0) {
            Timer tm(BWD_LU);
            backward_counter += reverse_local_update_linear_dht_topk(node_id, graph, lowest_rmax, in_backward,
                                                                     in_next_backward, backward_from);
            ppr_bi.insert(node_id, multi_bwd_idx_p[node_id]);
            for (auto item:multi_bwd_idx_r[node_id]) {
                int node = item.first;
                double residual = item.second;
                if (rw_counter.occur[node] == node_id) {
                    ppr_bi[node_id] += rw_counter[node] * 1.0 / num_s_rw_real * residual;
                }
            }
        } else {
            if (rw_counter.occur[node_id] == node_id) {
                ppr_bi.insert(node_id, rw_counter[node_id] / 1.0 / num_s_rw_real);
            }
        }
    }

    for (int j = 0; j < ppr.occur.m_num; ++j) {
        int node = ppr.occur[j];
        assert(ppr.exist(node));
        if (ppr[node] < 0) {
            cout << "!!!!!" << node << ppr[node] << endl;
        }
    }
    //INFO(real_total_rw);
    if (config.delta < threshold) {
        INFO(candidate.size());
        set_ppr_bounds(graph, check_rsum, real_total_rw);
        if (run_bippr) set_ppr_self_bounds(graph, candidate);
        //set_dht_bounds(candidate);
    }

    INFO(counter);
    INFO(num_total_rw - old_num_total_rw);
    INFO(backward_counter, candidate.size(), 1 / config2.rmax, config2.epsilon, config2.delta);

    cout << "ratio of bw and ra:\t" << backward_counter / fwd_idx.second.occur.m_num / config2.omega << endl;
    cout << "ratio of bw and true ra:\t" << backward_counter * 1.0 / (num_total_rw - old_num_total_rw) << endl;
    if (!candidate.empty()) {
        cout << "ratio of bw and esti bw:\t" << backward_counter / (candidate.size() / config2.rmax) << endl;
        cout << "ratio of ra and esti ra:\t" << (num_total_rw - old_num_total_rw) / candidate.size() / config2.omega
             << endl;
    }
    cout << "ratio of esti bw and esti ra:\t" << (1 / config2.rmax) / config2.omega << endl;
    num_total_bi += backward_counter;
#ifdef CHECK_PPR_VALUES
    display_ppr();
#endif
    return run_bippr;
}

void bippr_query_topk(int v, const Graph &graph) {
    Timer timer(0);

    ppr.clean();
    rw_counter.clean();

    {
        Timer tm(RONDOM_WALK);
        num_total_rw += config.omega;
        for (unsigned long i = 0; i < config.omega; i++) {
            int destination = random_walk(v, graph);
            if (rw_counter.notexist(destination)) {
                rw_counter.insert(destination, 1);
            } else {
                rw_counter[destination] += 1;
            }
        }
    }

    if (config.rmax < 1.0) {
        Timer tm(BWD_LU);
        for (int i = 0; i < graph.n; i++) {
            reverse_local_update_linear(i, graph);
            if ((!bwd_idx.first.exist(v) || 0 == bwd_idx.first[v]) && 0 == bwd_idx.second.occur.m_num) {
                continue;
            }

            if (bwd_idx.first.exist(v) && bwd_idx.first[v] > 0)
                ppr.insert(i, bwd_idx.first[v]);

            for (long j = 0; j < bwd_idx.second.occur.m_num; j++) {
                int nodeid = bwd_idx.second.occur[j];
                double residual = bwd_idx.second[nodeid];
                int occur;
                if (!rw_counter.exist(nodeid)) {
                    occur = 0;
                } else {
                    occur = rw_counter[nodeid];
                }

                if (occur > 0) {
                    if (!ppr.exist(i)) {
                        ppr.insert(i, occur * residual / config.omega);
                    } else {
                        ppr[i] += occur * residual / config.omega;
                    }
                }
            }
        }
    } else {
        int node_id;
        for (long i = 0; i < rw_counter.occur.m_num; i++) {
            node_id = rw_counter.occur[i];
            if (rw_counter[node_id] > 0) {
                if (!ppr.exist(node_id)) {
                    ppr.insert(node_id, rw_counter[node_id] * 1.0 / config.omega);
                } else {
                    ppr[node_id] = rw_counter[node_id] * 1.0 / config.omega;
                }
            }
        }
    }
}

void hubppr_query(int s, const Graph &graph) {}

void compute_ppr_with_fwdidx(const Graph &graph, double check_rsum) {
    ppr.reset_zero_values();
    //先取出P
    int node_id;
    double reserve;
    for (long i = 0; i < fwd_idx.first.occur.m_num; i++) {
        node_id = fwd_idx.first.occur[i];
        reserve = fwd_idx.first[node_id];
        ppr[node_id] = reserve;
    }

    // INFO("rsum is:", check_rsum);
    if (check_rsum == 0.0)
        return;
    INFO(check_rsum);
    unsigned long long num_random_walk = config.omega * check_rsum;//这里的omega并不是真正的Omega，num_random_walk才是
    INFO(num_random_walk);
    // INFO(num_random_walk);
    //num_total_rw += num_random_walk;
    int real_random_walk_f = 0;

    {
        Timer timer(RONDOM_WALK); //both rand-walk and source distribution are related with num_random_walk
        for (long i = 0; i < fwd_idx.second.occur.m_num; i++) {
            int source = fwd_idx.second.occur[i];
            double residual = fwd_idx.second[source];
            unsigned long num_s_rw = ceil(residual / check_rsum * num_random_walk);
            double a_s = residual / check_rsum * num_random_walk / num_s_rw;

            double ppr_incre = a_s * check_rsum / num_random_walk;

            num_total_rw += num_s_rw;
            real_random_walk_f += num_s_rw;
            for (unsigned long j = 0; j < num_s_rw; j++) {
                int des = random_walk(source, graph);
                ppr[des] += ppr_incre;
            }
        }
    }
    INFO(real_random_walk_f);
}

int compute_ppr_with_fwdidx_topk_with_bound_alias(const Graph &graph, double check_rsum) {
    compute_ppr_with_reserve();

    if (check_rsum == 0.0)
        return 0;

    long num_random_walk = ceil(config.omega * check_rsum);
    long real_num_rand_walk = num_random_walk;
    int count_rw_fora_iter = 0;
    double ppr_incre = check_rsum / num_random_walk;
    {
        Timer timer(RONDOM_WALK); //both rand-walk and source distribution are related with num_random_walk

        //构建 alias
        //先取出非零的r来
        vector<int> smaller, larger;
        unordered_map<int, double> prob;
        vector<pair<int, int>> alias;
        {
            Timer timer1(999);
            prob.reserve(fwd_idx.second.occur.m_num);
            smaller.reserve(fwd_idx.second.occur.m_num);
            larger.reserve(fwd_idx.second.occur.m_num);
            alias.reserve(fwd_idx.second.occur.m_num);
            for (int k = 0; k < fwd_idx.second.occur.m_num; ++k) {
                int node = fwd_idx.second.occur[k];
                if (fwd_idx.second[node] > 0) {
                    prob.insert(MP(node, fwd_idx.second[node]));
                }
            }
            int total = prob.size();
            for (auto &item:prob) {
                item.second *= total;
                if (item.second < 1.0) {
                    smaller.emplace_back(item.first);
                } else if (item.second > 1.0) {
                    larger.emplace_back(item.first);
                } else {
                    alias.emplace_back(MP(item.first, item.first));
                }
            }
            while (!smaller.empty() && !larger.empty()) {
                int small = smaller.back();
                smaller.pop_back();
                int large = larger.back();
                larger.pop_back();
                alias.emplace_back(MP(small, large));
                prob[large] -= (1 - prob[small]);
                if (prob[large] < 1) {
                    smaller.emplace_back(large);
                } else if (prob[large] > 1) {
                    larger.emplace_back(large);
                } else {
                    alias.emplace_back(MP(large, large));
                }
            }
        }
        for (int l = 0; l < num_random_walk; ++l) {
            int tmp = int(1.0 * rand() / RAND_MAX * alias.size());//随机取一列
            double seed = 1.0 * rand() / RAND_MAX;
            int source = -1;
            if (seed < prob[alias[tmp].first]) {
                source = alias[tmp].first;
            } else {
                source = alias[tmp].second;
            }
            int des = random_walk(source, graph);
            if (!ppr.exist(des))
                ppr.insert(des, ppr_incre);
            else
                ppr[des] += ppr_incre;
        }
    }

    INFO(real_num_rand_walk, count_rw_fora_iter);
    if (config.delta < threshold) {
        set_ppr_bounds(graph, check_rsum, real_num_rand_walk);
    } else {
        zero_ppr_upper_bound = calculate_lambda(check_rsum, config.pfail, zero_ppr_upper_bound, real_num_rand_walk);
    }
    num_total_rw += real_num_rand_walk;
    return real_num_rand_walk;
}

int compute_ppr_with_fwdidx_topk_with_bound(const Graph &graph, double check_rsum,
                                            unordered_map<int, vector<int>> &rw_saver) {
    compute_ppr_with_reserve();

    if (check_rsum == 0.0)
        return 0;

    long num_random_walk = config.omega * check_rsum;
    long real_num_rand_walk = 0;
    int count_rw_fora_iter = 0, map_counter = 0;
    {
        Timer timer(RONDOM_WALK); //both rand-walk and source distribution are related with num_random_walk

        //Timer tm(SOURCE_DIST);
        { //rand walk online
            for (long i = 0; i < fwd_idx.second.occur.m_num; i++) {
                int source = fwd_idx.second.occur[i];
                double residual = fwd_idx.second[source];
                long num_s_rw = ceil(residual / check_rsum * num_random_walk);
                double a_s = residual / check_rsum * num_random_walk / num_s_rw;

                real_num_rand_walk += num_s_rw;
                num_total_rw += num_s_rw;
                double ppr_incre = a_s * check_rsum / num_random_walk;

                for (long j = 0; j < num_s_rw; j++) {
                    int des = random_walk(source, graph);
                    if (!ppr.exist(des))
                        ppr.insert(des, ppr_incre);
                    else
                        ppr[des] += ppr_incre;

                }
            }
        }
    }
    if (config.delta < threshold)
        set_ppr_bounds(graph, check_rsum, real_num_rand_walk);
    return real_num_rand_walk;
}

int compute_ppr_with_fwdidx_topk_with_bound_with_idx(const Graph &graph, double check_rsum,
                                            unordered_map<int, vector<int>> &rw_saver) {
    compute_ppr_with_reserve();

    if (check_rsum == 0.0)
        return 0;

    long num_random_walk = config.omega * check_rsum;
    long real_num_rand_walk = 0;
    int count_rw_fora_iter = 0, map_counter = 0;
    {
        Timer timer(RONDOM_WALK); //both rand-walk and source distribution are related with num_random_walk

        //Timer tm(SOURCE_DIST);
        { //rand walk online
            for (long i = 0; i < fwd_idx.second.occur.m_num; i++) {
                int source = fwd_idx.second.occur[i];
                double residual = fwd_idx.second[source];
                long num_s_rw = ceil(residual / check_rsum * num_random_walk);
                double a_s = residual / check_rsum * num_random_walk / num_s_rw;

                real_num_rand_walk += num_s_rw;
                num_total_rw += num_s_rw;
                double ppr_incre = a_s * check_rsum / num_random_walk;
                int index_size=rw_index[source].size();
                if (num_s_rw>index_size){
                    for (int j = 0; j < index_size; ++j) {
                        int des =rw_index[source][j];
                        if (!ppr.exist(des))
                            ppr.insert(des, ppr_incre);
                        else
                            ppr[des] += ppr_incre;
                    }
                    for (int k = 0; k < num_s_rw - index_size; ++k) {
                        int des = random_walk(source, graph);
                        rw_index[source].emplace_back(des);
                        if (!ppr.exist(des))
                            ppr.insert(des, ppr_incre);
                        else
                            ppr[des] += ppr_incre;
                    }
                }else{
                    for (int j = 0; j < num_s_rw; ++j) {
                        int des = rw_index[source][j];
                        if (!ppr.exist(des))
                            ppr.insert(des, ppr_incre);
                        else
                            ppr[des] += ppr_incre;
                    }
                }
            }
        }
    }
    if (config.delta < threshold)
        set_ppr_bounds(graph, check_rsum, real_num_rand_walk);
    return real_num_rand_walk;
}
void fora_bippr_query(int v, const Graph &graph, double raw_epsilon) {
    Timer timer(FB_QUERY);
    double rsum = 1.0, ratio = graph.n;//sqrt(config.alpha);
    fora_bippr_setting(graph.n, graph.m, ratio, raw_epsilon);
    static vector<int> forward_from;
    forward_from.reserve(graph.n);
    forward_from.push_back(v);
    //fora_bippr_setting(graph.n, graph.m, 1,raw_epsilon);
    {
        Timer timer(FORA_QUERY);
        //初始时，设置两者一样，然后第一次检测大小关系，
        {
            Timer timer(FWD_LU);
            fwd_idx.first.clean();  //reserve
            fwd_idx.second.clean();  //residual
            fwd_idx.second.insert(v, rsum);
            long forward_push_num = 0;
            bool flag= true,first= true;
            do {
                cout << endl;
                //display_setting();
                if (!first)
                    flag=check_cost(rsum, ratio, graph.n, graph.m, forward_push_num, raw_epsilon);
                first= false;
                forward_push_num += forward_local_update_linear_topk_dht2(v, graph, rsum, config.rmax, 0, forward_from);
            } while (flag);
            /*
            //display_setting();
            cout << "epsilon:\t" << config.epsilon << "\t" << config2.epsilon << endl;
            cout << "omega:\t" << config.omega << "\t" << config2.omega << endl;
            cout << (1 + config.epsilon) / (1 - config2.epsilon) << endl;
            do {
                cout << endl;
            } while (false);
            //(check_cost(rsum, ratio, graph.n, graph.m));
            */
            //forward_local_update_linear(v, graph, rsum, config.rmax); //forward propagation, obtain reserve and residual
        }
        bippr_query_with_fora(graph, rsum);
        //compute_ppr_with_fwdidx(graph, rsum);
        //bippr_query_fb_raw(graph, config.delta);
    }
    INFO(config2.omega * fwd_idx.second.occur.m_num);
    int num_rw_bi = num_total_rw;
    INFO(num_rw_bi);
    INFO(graph.m * config.rmax * config.omega);
    INFO(rsum * config.omega);
}

/*
 *
    ///config2的delta在后面确定！
    ///zero_ppr_upper_bound = 1.0;
    另外不用初始化dht的上下界。
 */
void fora_bippr_query_topk(int v, const Graph &graph) {
    //ppr fwd_idx.first fwd_idx.second的occur都是有效的
    Timer timer(0);
    const static double min_delta = config.alpha / graph.n;
    INFO(min_delta);
    const static double init_delta = 1.0 / 4;
    const static double min_epsilon = sqrt(1.0 / graph.n / config.alpha);
    const static double raw_epsilon = config.epsilon;
    threshold = (1.0 - config.ppr_decay_alpha) / pow(500, config.ppr_decay_alpha) /
                pow(graph.n, 1 - config.ppr_decay_alpha);//?
    const static double new_pfail = 1.0 / graph.n / graph.n / log(graph.n);
//sqrt(0.2*281904/0.00010715135)/500
    config.pfail = new_pfail;  // log(1/pfail) -> log(1*n/pfail)
    config.pfail /= 2;
    config2 = config;
    //config.epsilon = config.epsilon / (2 + config.epsilon);//(1+new_epsilon)/(1-new_epsilon)<=1+epsilon
    double ratio_f_b = sqrt(config.alpha * graph.n / threshold) / config.k;
    config2.delta = config.alpha;
    config.delta = init_delta;

    const static double lowest_delta_rmax =
            config.epsilon * sqrt(min_delta / 3 / graph.n / log(2 / new_pfail));//更新后的r_max
    const static double lowest_delta_rmax_2 =
            min_epsilon * sqrt(config2.delta / 3 / log(2.0 / new_pfail));

    double rsum = 1.0;
    //迭代起始点
    static vector<int> forward_from;
    forward_from.clear();
    forward_from.reserve(graph.n);
    forward_from.push_back(v);
    unordered_map<int, vector<int>> backward_from;
    //p和r等
    fwd_idx.first.clean();  //reserve
    fwd_idx.second.clean();  //residual
    fwd_idx.second.insert(v, rsum);
    multi_bwd_idx_p.clear();
    multi_bwd_idx_r.clear();
    ///config2的delta在后面确定！
    ///zero_ppr_upper_bound = 1.0;
    //先过一遍拿到上下限
    upper_bounds.reset_one_values();//不可使用上下界的occur遍历了
    lower_bounds.reset_zero_values();
    init_bounds_self(graph);
    upper_bounds_dht.reset_one_values();
    lower_bounds_dht.reset_zero_values();
    //set<int> candidate;
    unordered_map<int, bool> candidate;
    candidate.clear();
    display_setting();
    bool run_fora = true;
    unordered_map<int, vector<int>> rw_saver;
    //调用fora，但终止条件不同
    num_total_fo = 0;
    num_total_bi = 0;
    int init_rw_num = num_total_rw;
    while (config.delta >= min_delta) {
        double old_f_rmax = config.rmax, old_b_rmax = config2.rmax, forward_counter, backward_counter, real_num_rand_walk;
        int old_candidate_size = candidate.size();
        fora_bippr_setting(graph.n, graph.m, 1, 1, true);
        if (run_fora) {
            num_iter_topk++;
            {
                Timer timer(FWD_LU);
                forward_counter = forward_local_update_linear_topk_dht2(v, graph, rsum, config.rmax, lowest_delta_rmax,
                                                                        forward_from); //forward propagation, obtain reserve and residual
            }
            real_num_rand_walk = compute_ppr_with_fwdidx_topk_with_bound_with_idx(graph, rsum, rw_saver);
        } else {
            backward_counter = bippr_query_candidate_with_idx(graph, candidate, lowest_delta_rmax_2, backward_from);
        }
        if (config.delta < threshold) {
            set_dht_bounds(candidate);
        }
        INFO(config.delta, config.rmax, config.omega, fwd_idx.second.occur.m_size, candidate.size());
        //compute_ppr_with_fwdidx_topk_with_bound(graph, rsum);
        if (if_stop_dht(candidate, raw_epsilon, min_delta) ||
            (config.delta <= min_delta && config2.epsilon <= min_epsilon)) {

            dht.clean();
            for (auto item:candidate) {
                int node = item.first;
                if(ppr_bi[node]<=0){
                    INFO(node);
                }
                dht.insert(node, ppr[node] / ppr_bi[node]);
            }
            break;
        } else if (config.delta > min_delta && config2.epsilon > min_epsilon) {
            //计算两者的复杂度，给出config2的epsilon

            run_fora = test_run_fora(candidate.size(), graph.m, graph.n, forward_counter, real_num_rand_walk);
            if (run_fora) {
                config.delta = max(min_delta, config.delta / 2.0);  // otherwise, reduce delta to delta/2
            } else {
                config2.epsilon /= 2;
            }
        } else if (config.delta > min_delta) {
            run_fora = test_run_fora(candidate.size(), graph.m, graph.n, forward_counter, real_num_rand_walk);
            if (run_fora == false) {
                dht.clean();
                for (auto item:candidate) {
                    int node = item.first;
                    if(ppr_bi[node]<=0){
                        INFO(node);
                    }
                    dht.insert(node, ppr[node] / ppr_bi[node]);
                }
                break;
            }
            config.delta = max(min_delta, config.delta / 2.0);  // otherwise, reduce delta to delta/2
        } else {
            if (config2.epsilon == raw_epsilon) {
                config2.epsilon = max(min_epsilon, config2.epsilon / 2);
                run_fora = false;
            } else {
                //如果变化还很大，就继续吧
                if (old_candidate_size > candidate.size() * 1.01) {
                    config2.epsilon = max(min_epsilon, config2.epsilon / 2);
                    run_fora = false;
                    continue;
                }
                run_fora = test_run_fora(candidate.size(), graph.m, graph.n, forward_counter, real_num_rand_walk);
                if (run_fora == true) {
                    dht.clean();
                    for (auto item:candidate) {
                        int node = item.first;
                        if(ppr_bi[node]<=0){
                            INFO(node);
                        }
                        dht.insert(node, ppr[node] / ppr_bi[node]);
                    }
                    break;
                } else {
                    config2.epsilon = max(min_epsilon, config2.epsilon / 2);
                }
            }
        }
    }
    INFO(num_total_rw - init_rw_num);
    return;

}


void fb_raw_query(int v, const Graph &graph, bool topk = false) {
    Timer tm(FBRAW_QUERY);
    double rsum = 1.0, threshold;
    display_setting();
    {
        Timer timer(FWD_LU);
        forward_local_update_linear(v, graph, rsum, config.rmax); //forward propagation, obtain reserve and residual
    }
    // compute_ppr_with_fwdidx(graph);
    compute_ppr_with_fwdidx(graph, rsum);
    if (topk) {
        static vector<double> temp_ppr;
        temp_ppr.clear();
        temp_ppr.resize(graph.n);
        int size = 0;
        for (int i = 0; i < graph.n; i++) {
            if (ppr.m_data[i] > 0)
                temp_ppr[size++] = ppr.m_data[i];
        }
        nth_element(temp_ppr.begin(), temp_ppr.begin() + config.k - 1, temp_ppr.end(), cmp);
        threshold = temp_ppr[config.k - 1] * config.alpha;
    } else {
        threshold = config.delta;
    }
    bippr_query_fb_raw(graph, threshold);
}

void fora_query_basic(int v, const Graph &graph) {
    Timer timer(FORA_QUERY);
    double rsum = 1.0;

    {
        Timer timer(FWD_LU);
        forward_local_update_linear(v, graph, rsum, config.rmax); //forward propagation, obtain reserve and residual
    }
    // compute_ppr_with_fwdidx(graph);
    compute_ppr_with_fwdidx(graph, rsum);

#ifdef PRINT_PPR_VALUES
    ofstream fout(".//PPR/"+config.graph_alias+std::to_string(v) +".txt");
    if (fout.is_open()) {
        for(int i=0; i< ppr.occur.m_num; i++){
            if(ppr[ppr.occur[i]]>config.delta){
                fout << ppr.occur[i] << "->" << ppr[ ppr.occur[i] ] << endl;
            }
        }
        fout.close();
    }
#endif

#ifdef CHECK_PPR_VALUES
    display_ppr();
#endif
}

void fora_query_topk_with_bound(int v, const Graph &graph) {
    Timer timer(0);
    const static double min_delta = 1.0 / graph.n;
    const static double init_delta = 1.0 / 4;
    threshold = (1.0 - config.ppr_decay_alpha) / pow(500, config.ppr_decay_alpha) /
                pow(graph.n, 1 - config.ppr_decay_alpha);//?

    const static double new_pfail = 1.0 / graph.n / graph.n / log(graph.n);

    config.pfail = new_pfail;  // log(1/pfail) -> log(1*n/pfail)
    config.delta = init_delta;

    const static double lowest_delta_rmax =
            config.epsilon * sqrt(min_delta / 3 / graph.m / log(2 / new_pfail));//更新后的r_max

    double rsum = 1.0;

    static vector<int> forward_from;
    forward_from.clear();
    forward_from.reserve(graph.n);
    forward_from.push_back(v);

    fwd_idx.first.clean();  //reserve
    fwd_idx.second.clean();  //residual
    fwd_idx.second.insert(v, rsum);

    zero_ppr_upper_bound = 1.0;

    if (config.with_rw_idx)
        rw_counter.reset_zero_values(); //store the pointers of each node's used rand-walk idxs

    // for delta: try value from 1/4 to 1/n
    int iteration = 0;
    upper_bounds.reset_one_values();
    lower_bounds.reset_zero_values();
    unordered_map<int, vector<int>> rw_saver;
    while (config.delta >= min_delta) {
        fora_setting(graph.n, graph.m);
        num_iter_topk++;

        {
            Timer timer(FWD_LU);
            forward_local_update_linear_topk(v, graph, rsum, config.rmax, lowest_delta_rmax,
                                             forward_from); //forward propagation, obtain reserve and residual
        }

        compute_ppr_with_fwdidx_topk_with_bound(graph, rsum, rw_saver);
        if (if_stop() || config.delta <= min_delta) {
            break;
        } else
            config.delta = max(min_delta, config.delta / 2.0);  // otherwise, reduce delta to delta/2
    }
}


iMap<int> updated_pprs;

void hubppr_query_topk_martingale(int s, const Graph &graph) {}

void get_topk(int v, Graph &graph) {
    display_setting();
    if (config.algo == MC) {
        montecarlo_query_topk(v, graph);
        topk_ppr();
    } else if (config.algo == BIPPR) {
        bippr_query_topk(v, graph);
        topk_ppr();
    } else if (config.algo == FORA) {
        fora_query_topk_with_bound(v, graph);
        topk_ppr();
    } else if (config.algo == FWDPUSH) {
        Timer timer(0);
        double rsum = 1;

        {
            Timer timer(FWD_LU);
            forward_local_update_linear(v, graph, rsum, config.rmax);
        }
        compute_ppr_with_reserve();
        topk_ppr();
    } else if (config.algo == HUBPPR) {
        Timer timer(0);
        hubppr_query_topk_martingale(v, graph);
    }

    // not FORA, so it's single source
    // no need to change k to run again
    // check top-k results for different k
    if (config.algo != FORA && config.algo != HUBPPR) {
        compute_precision_for_dif_k(v);
    }

    compute_precision(v);

#ifdef CHECK_TOP_K_PPR
    vector<pair<int, double>>& exact_result = exact_topk_pprs[v];
    INFO("query node:", v);
    for(int i=0; i<topk_pprs.size(); i++){
        cout << "Estimated k-th node: " << topk_pprs[i].first << " PPR score: " << topk_pprs[i].second << " " << map_lower_bounds[topk_pprs[i].first].first<< " " << map_lower_bounds[topk_pprs[i].first].second
             <<" Exact k-th node: " << exact_result[i].first << " PPR score: " << exact_result[i].second << endl;
    }
#endif
}

void get_topk2(int v, Graph &graph) {

    display_setting();
    if (config.algo == MC_DHT) {
        montecarlo_query_dht_topk(v, graph);
        topk_dht();
        //INFO(topk_dhts);
    } else if (config.algo == DNE) {///DNE在数据集比较大的时候就变得缓慢
        Timer timer(0);
        dne_query(v, graph);
        //topk_dht();
    } else if (config.algo == FB_RAW) {
        //先使用FORA，然后使用1/alpha ~ 1/(2-alpha)这个范围，计算这些点的BiPPR
        fb_raw_query(v, graph, true);
        //fb_raw_query_topk(v, graph);
        topk_dht();
    } else if (config.algo == FB) {
        rw_index.clear();
        fora_bippr_query_topk(v, graph);
        topk_dht();
        topk_ppr();
        //display_topk_dht();
    }

    // not FORA, so it's single source
    // no need to change k to run again
    /// check top-k results for different k
    if (config.algo == MC_DHT) {
        compute_precision_for_dif_k_dht(v);
    }

    //compute_precision(v);
    compute_precision_dht(v);

#ifdef CHECK_TOP_K_PPR
    vector<pair<int, double>>& exact_result = exact_topk_pprs[v];
    INFO("query node:", v);
    for(int i=0; i<topk_pprs.size(); i++){
        cout << "Estimated k-th node: " << topk_pprs[i].first << " PPR score: " << topk_pprs[i].second << " " << map_lower_bounds[topk_pprs[i].first].first<< " " << map_lower_bounds[topk_pprs[i].first].second
             <<" Exact k-th node: " << exact_result[i].first << " PPR score: " << exact_result[i].second << endl;
    }
#endif

}
void fwd_power_iteration(const Graph &graph, int start, unordered_map<int, double> &map_ppr) {
    static thread_local unordered_map<int, double> map_residual;
    map_residual[start] = 1.0;

    int num_iter = 0;
    double rsum = 1.0;
    while (num_iter < config.max_iter_num) {
        num_iter++;
        // INFO(num_iter, rsum);
        vector<pair<int, double> > pairs(map_residual.begin(), map_residual.end());
        map_residual.clear();
        for (const auto &p: pairs) {
            if (p.second > 0) {
                map_ppr[p.first] += config.alpha * p.second;
                int out_deg = graph.g[p.first].size();

                double remain_residual = (1 - config.alpha) * p.second;
                rsum -= config.alpha * p.second;
                if (out_deg == 0) {
                    map_residual[start] += remain_residual;
                } else {
                    double avg_push_residual = remain_residual / out_deg;
                    for (int next : graph.g[p.first]) {
                        map_residual[next] += avg_push_residual;
                    }
                }
            }
        }
        pairs.clear();
    }
    map_residual.clear();
}

void fwd_power_iteration_self(const Graph &graph, int start, Bwdidx &bwd_idx_th, vector<int> &idx, vector<int> &q) {
    //static thread_local unordered_map<int, double> map_residual;
    int pointer_q = 0;
    int left = 0;
    double myeps = 1.0 / 10000000000;
    q[pointer_q++] = start;
    bwd_idx_th.second.occur[start] = start;
    bwd_idx_th.second[start] = 1;

    idx[start] = start;
    while (left != pointer_q) {
        int v = q[left++];
        left %= graph.n;
        idx[v] = -1;
        if (bwd_idx_th.second[v] < myeps)
            break;

        if (bwd_idx_th.first.occur[v] != start) {
            bwd_idx_th.first.occur[v] = start;
            bwd_idx_th.first[v] = bwd_idx_th.second[v] * config2.alpha;
        } else
            bwd_idx_th.first[v] += bwd_idx_th.second[v] * config.alpha;

        double residual = (1 - config2.alpha) * bwd_idx_th.second[v];
        bwd_idx_th.second[v] = 0;
        if (graph.gr[v].size() > 0) {
            for (int next : graph.gr[v]) {
                int cnt = graph.g[next].size();
                if (bwd_idx_th.second.occur[next] != start) {
                    bwd_idx_th.second.occur[next] = start;
                    bwd_idx_th.second[next] = residual / cnt;
                } else
                    bwd_idx_th.second[next] += residual / cnt;

                if (bwd_idx_th.second[next] > myeps && idx[next] != start) {
                    // put next into q if next is not in q
                    idx[next] = start;//(int) q.size();
                    //q.push_back(next);
                    q[pointer_q++] = next;
                    pointer_q %= graph.n;
                }
            }
        }
    }
}

void multi_power_iter(const Graph &graph, const vector<int> &source,
                      unordered_map<int, vector<pair<int, double>>> &map_topk_ppr) {
    static thread_local unordered_map<int, double> map_ppr;
    for (int start: source) {
        fwd_power_iteration(graph, start, map_ppr);

        vector<pair<int, double>> temp_top_ppr(config.k);
        partial_sort_copy(map_ppr.begin(), map_ppr.end(), temp_top_ppr.begin(), temp_top_ppr.end(),
                          [](pair<int, double> const &l, pair<int, double> const &r) { return l.second > r.second; });

        map_ppr.clear();
        map_topk_ppr[start] = temp_top_ppr;
        INFO(start);
    }
}

void multi_power_iter_self(const Graph &graph, const vector<int> &source,
                           unordered_map<int, double> &map_self_ppr) {
    //static thread_local unordered_map<int, double> map_ppr;
    static int count = 0;
    static thread_local Bwdidx bwd_idx_th;
    bwd_idx_th.first.initialize(graph.n);
    bwd_idx_th.second.initialize(graph.n);
    fill(bwd_idx_th.first.occur.m_data, bwd_idx_th.first.occur.m_data + graph.n, -1);
    fill(bwd_idx_th.second.occur.m_data, bwd_idx_th.second.occur.m_data + graph.n, -1);
    static thread_local vector<int> idx(graph.n, -1), q(graph.n);
    for (int start: source) {
        count++;
        if (count % 1000 == 0) INFO(count);
        fwd_power_iteration_self(graph, start, bwd_idx_th, idx, q);
        std::mutex g_mutex;
        /*
        g_mutex.lock();
        INFO(start);
        INFO(bwd_idx_th.first[start]);
        g_mutex.unlock();*/
        map_self_ppr[start] = bwd_idx_th.first[start];
    }
}

void gen_exact_self(const Graph &graph) {
    // config.epsilon = 0.5;
    // montecarlo_setting();
    load_exact_topk_ppr();
    map<int, double> ppr_self_old = load_exact_self_ppr();
    //load_exact_self_ppr();
    //map<int ,double> ppf_self=load_exact_self_ppr();
    //vector<double> ppf_self=load_exact_self_ppr_vec();
    /*
    unordered_map<int,double> ppr_self_old;
    for(auto item1:ppf_self){
        ppr_self_old[item1.first]=item1.second;
    }*/
    double min_rmax = 1.0 / 2000;
    set<int> candidate_node;
    bwd_idx.first.initialize(graph.n);
    bwd_idx.second.initialize(graph.n);
    vector<int> idx(graph.n, -1), node_with_r(graph.n), q(graph.n);
    int pointer_r = 0, max_candi_num = ceil(config.k / (2 - config.alpha) / config.alpha);
    int cur = 0;
    for (auto item:exact_topk_pprs) {
        fill(bwd_idx.first.occur.m_data, bwd_idx.first.occur.m_data + graph.n, -1);
        fill(bwd_idx.second.occur.m_data, bwd_idx.second.occur.m_data + graph.n, -1);
        if (++cur > config.query_size)break;
        int source_node = item.first;
        INFO(source_node);
        unordered_map<int, double> upper_bound, lower_bound;
        unordered_map<int, double> upper_bound_self, lower_bound_self;
        vector<int> candidate_s;
        for (int j = 0; j < max_candi_num; ++j) {
            int node = item.second[j].first;
            double ppr = item.second[j].second;
            config2.rmax = min_rmax;
            //reverse_local_update_linear(node, graph);
            reverse_local_update_linear_dht(node, graph, idx, node_with_r, pointer_r, q);
            upper_bound_self[node] = bwd_idx.first[node] + min_rmax;
            lower_bound_self[node] = bwd_idx.first[node];
            upper_bound[node] = ppr / bwd_idx.first[node];
            lower_bound[node] = ppr / (bwd_idx.first[node] + min_rmax);
            //INFO(node,upper_bound[node],lower_bound[node],ppr,bwd_idx.first[node]);
        }
        vector<double> tmp_dht(lower_bound.size());
        int cur = 0;
        for (auto item:lower_bound) {
            tmp_dht[cur++] = item.second;
        }
        nth_element(tmp_dht.begin(), tmp_dht.begin() + config.k - 1, tmp_dht.end(), cmp);
        cur = 0;
        //INFO(tmp_dht[config.k - 1]);
        for (auto item:upper_bound) {
            if (item.second >= tmp_dht[config.k - 1]) {
                if (ppr_self_old.find(item.first) == ppr_self_old.end() ||
                    ppr_self_old.at(item.first) > upper_bound_self[item.first] ||
                    ppr_self_old.at(item.first) < lower_bound_self[item.first]) {
                    cur++;
                    cout << item.first << "\t";
                    candidate_node.emplace(item.first);
                }
            }
        }
        INFO(cur);
    }
    vector<int> queries;
    queries.assign(candidate_node.begin(), candidate_node.end());
    unsigned int query_size = queries.size();
    INFO(queries.size());
    split_line();
    // montecarlo_setting();

    unsigned NUM_CORES = std::thread::hardware_concurrency() - 2;
    assert(NUM_CORES >= 2);

    int num_thread = min(query_size, NUM_CORES);
    int avg_queries_per_thread = query_size / num_thread;

    vector<vector<int>> source_for_all_core(num_thread);
    vector<unordered_map<int, double >> ppv_self_for_all_core(num_thread);

    for (int tid = 0; tid < num_thread; tid++) {
        int s = tid * avg_queries_per_thread;
        int t = s + avg_queries_per_thread;

        if (tid == num_thread - 1)
            t += query_size % num_thread;

        for (; s < t; s++) {
            // cout << s+1 <<". source node:" << queries[s] << endl;
            source_for_all_core[tid].push_back(queries[s]);
        }
    }


    {
        Timer timer(PI_QUERY_SELF);
        INFO("power itrating...");
        std::vector<std::future<void> > futures(num_thread);
        for (int tid = 0; tid < num_thread; tid++) {
            futures[tid] = std::async(std::launch::async, multi_power_iter_self, std::ref(graph),
                                      std::ref(source_for_all_core[tid]), std::ref(ppv_self_for_all_core[tid]));
        }
        std::for_each(futures.begin(), futures.end(), std::mem_fn(&std::future<void>::wait));
    }

    // cout << "average iter times:" << num_iter_topk/query_size << endl;
    cout << "average generation time (s): " << Timer::used(PI_QUERY_SELF) * 1.0 / query_size << endl;

    INFO("combine results...");
    //map<int, double> ppr_self;
    for (int tid = 0; tid < num_thread; tid++) {
        for (auto &ppv: ppv_self_for_all_core[tid]) {
            //exact_topk_pprs.insert(ppv);
            ppr_self_old[ppv.first] = ppv.second;
        }
    }
    save_self_ppr(ppr_self_old);
    //save_exact_topk_ppr();
}

set<int> gen_exact_topk(const Graph &graph) {
    // config.epsilon = 0.5;
    // montecarlo_setting();

    vector<int> queries;
    load_ss_query(queries);
    INFO(queries.size());
    unsigned int query_size = queries.size();
    query_size = min(query_size, config.query_size);
    INFO(query_size);

    assert(config.k < graph.n - 1);
    assert(config.k > 1);
    INFO(config.k);

    split_line();
    // montecarlo_setting();

    unsigned NUM_CORES = std::thread::hardware_concurrency() - 1;
    assert(NUM_CORES >= 2);

    int num_thread = min(query_size, NUM_CORES);
    int avg_queries_per_thread = query_size / num_thread;

    vector<vector<int>> source_for_all_core(num_thread);
    vector<unordered_map<int, vector<pair<int, double>>>> ppv_for_all_core(num_thread);

    for (int tid = 0; tid < num_thread; tid++) {
        int s = tid * avg_queries_per_thread;
        int t = s + avg_queries_per_thread;

        if (tid == num_thread - 1)
            t += query_size % num_thread;

        for (; s < t; s++) {
            // cout << s+1 <<". source node:" << queries[s] << endl;
            source_for_all_core[tid].push_back(queries[s]);
        }
    }


    {
        Timer timer(PI_QUERY);
        INFO("power itrating...");
        std::vector<std::future<void> > futures(num_thread);
        for (int tid = 0; tid < num_thread; tid++) {
            futures[tid] = std::async(std::launch::async, multi_power_iter, std::ref(graph),
                                      std::ref(source_for_all_core[tid]), std::ref(ppv_for_all_core[tid]));
        }
        std::for_each(futures.begin(), futures.end(), std::mem_fn(&std::future<void>::wait));
    }

    // cout << "average iter times:" << num_iter_topk/query_size << endl;
    cout << "average generation time (s): " << Timer::used(PI_QUERY) * 1.0 / query_size << endl;

    INFO("combine results...");
    set<int> results;
    for (int tid = 0; tid < num_thread; tid++) {
        for (auto &ppv: ppv_for_all_core[tid]) {
            exact_topk_pprs.insert(ppv);
            for (auto item:ppv.second) {
                results.emplace(item.first);
            }
        }
        ppv_for_all_core[tid].clear();
    }

    save_exact_topk_ppr();
    return results;
}

void topk(Graph &graph) {
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
    }
}

void topk2(Graph &graph) {
    vector<int> queries;
    load_ss_query(queries);
    INFO(queries.size());
    unsigned int query_size = queries.size();
    query_size = min(query_size, config.query_size);
    vector<int> true_queries;
    int used_counter = 0;

    assert(config.k < graph.n - 1);
    assert(config.k > 1);
    INFO(config.k);

    split_line();

    load_exact_topk_ppr();
    map<int, double> ppr_self = load_exact_self_ppr();
    //INFO(ppr_self[62505]);

    bwd_idx.first.initialize(graph.n);
    bwd_idx.second.initialize(graph.n);
    config.rmax=0.0001;
    int cur = 0;
    for (auto item:exact_topk_pprs) {
        if (++cur > config.query_size)break;
        int source_node = item.first;
        true_queries.emplace_back(source_node);
        //INFO(cur);
        //INFO(source_node);
        unordered_map<int, double> dht;
        vector<pair<int, double>> topk_pprs = item.second;
        for (int k = 0; k < topk_pprs.size(); ++k) {
            int node = topk_pprs[k].first;

            if (ppr_self.find(node) != ppr_self.end()) {
                dht[node] = topk_pprs[k].second / ppr_self.at(node);
                //INFO(dht[node]);
            } else {
                //INFO(node);
                //reverse_local_update_linear(node,graph);
                //INFO(topk_pprs[k].second / bwd_idx.first[node]);
            }
        }
        vector<pair<int, double>> temp_top_dht(500);
        partial_sort_copy(dht.begin(), dht.end(), temp_top_dht.begin(), temp_top_dht.end(),
                          [](pair<int, double> const &l, pair<int, double> const &r) { return l.second > r.second; });
        exact_topk_dhts[source_node] = temp_top_dht;
        //INFO(temp_top_dht);
    }

    // not FORA, so it's single source
    // no need to change k to run again
    // check top-k results for different k
    if (config.algo == MC_DHT) {
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
    if (config.algo == MC_DHT) {
        rw_counter.initialize(graph.n);
        dht.initialize(graph.n);
        montecarlo_setting();
    } else if (config.algo == DNE) {
        dht.initialize(graph.n);
    } else if (config.algo == FB_RAW) {
        //先使用FORA，然后使用1/alpha ~ 1这个范围，计算这些点的BiPPR
        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);
        rw_counter.init_keys(graph.n);
        upper_bounds.init_keys(graph.n);
        lower_bounds.init_keys(graph.n);
        ppr.initialize(graph.n);
        //使用vector代替map，提前预设向量大小为节点大小
        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);
        bwd_idx.first.initialize(graph.n);
        bwd_idx.second.initialize(graph.n);
        double raw_epsilon = config.epsilon, ratio = 1;
        fb_raw_setting(graph.n, graph.m, ratio, raw_epsilon);
    } else if (config.algo == FB) {
        //先利用节点的度得到初步的范围，
        // 另外重复利用RW？
        // 然后维护真实TOPK，计算UB的nth-k，对topk LB进行过滤，在候选集比较小的时候，使用哈希表保存，另外比较成本，使用成本低的。
        fwd_idx.first.initialize(graph.n);//forward p
        fwd_idx.second.initialize(graph.n);//forward r
        rw_counter.init_keys(graph.n);//
        rw_bippr_counter.init_keys(graph.n);
        upper_bounds.init_keys(graph.n);
        lower_bounds.init_keys(graph.n);
        upper_bounds_self.init_keys(graph.n);
        upper_bounds_self_init.init_keys(graph.n);
        lower_bounds_self.init_keys(graph.n);
        upper_bounds_dht.init_keys(graph.n);
        lower_bounds_dht.init_keys(graph.n);
        ppr.initialize(graph.n);
        ppr_bi.initialize(graph.n);
        dht.initialize(graph.n);
        topk_filter.initialize(graph.n);
    }

    for (int i = 0; i < query_size; i++) {
        cout << i + 1 << ". source node:" << true_queries[i] << endl;
        get_topk2(true_queries[i], graph);
        if (!config.NDCG) continue;
        if (config.algo == MC_DHT) {
            compute_NDCG_for_dif_k_dht(true_queries[i], ppr_self, graph);
        } else if (config.algo != DNE) {
            compute_NDCG(true_queries[i], ppr_self, graph);
        }
        split_line();
    }

    cout << "average iter times:" << num_iter_topk / query_size << endl;
    display_time_usage(used_counter, query_size);
    set_result(graph, used_counter, query_size);

    //not FORA, so it's single source
    //no need to change k to run again
    // check top-k results for different k
    if (config.algo == MC_DHT) {
        display_precision_for_dif_k();
    }
}

void query_dht(Graph &graph) {
    INFO(config.algo);
    vector<int> queries;
    load_ss_query(queries);
    unsigned int query_size = queries.size();
    query_size = min(query_size, config.query_size);
    INFO(query_size);
    int used_counter = 0;

    // assert(config.rw_cost_ratio >= 0);
    // INFO(config.rw_cost_ratio);

    assert(config.rmax_scale >= 0);
    INFO(config.rmax_scale);

    ppr.initialize(graph.n);
    dht.init_keys(graph.n);
    // sfmt_init_gen_rand(&sfmtSeed , 95082);
    if (config.algo == MC_DHT) {
        montecarlo_setting();
        display_setting();
        used_counter = MC_DHT_QUERY;

        rw_counter.initialize(graph.n);

        for (int i = 0; i < query_size; i++) {
            cout << i + 1 << ". source node:" << queries[i] << endl;
            montecarlo_query_dht(queries[i], graph);
            split_line();
        }
    } else if (config.algo == FB_RAW) {
        used_counter = FBRAW_QUERY;
        double raw_epsilon = config.epsilon, ratio = 1;
        fb_raw_setting(graph.n, graph.m, ratio, raw_epsilon);
        display_setting();
        //使用vector代替map，提前预设向量大小为节点大小
        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);
        bwd_idx.first.initialize(graph.n);
        bwd_idx.second.initialize(graph.n);

        rw_counter.initialize(graph.n);
        for (int i = 0; i < query_size; i++) {
            cout << i + 1 << ". source node:" << queries[i] << endl;
            fb_raw_query(queries[i], graph);
            INFO(Timer::used(5));
            INFO(Timer::used(6));
            INFO(Timer::used(9));
            split_line();
        }

    } else if (config.algo == GI) {
        used_counter = GI_QUERY;
        for (int i = 0; i < query_size; i++) {
            cout << i + 1 << ". source node:" << queries[i] << endl;
            global_iteration_query(queries[i], graph);
            split_line();
        }
    } else if (config.algo == DNE) {
        used_counter = DNE_QUERY;
        for (int i = 0; i < query_size; i++) {
            cout << i + 1 << ". source node:" << queries[i] << endl;
            dne_query(queries[i], graph);
            INFO(Timer::used(111));
            INFO(Timer::used(DNE_QUERY));
            split_line();
        }
    } else if (config.algo == FB) {
        //fora epsilon? delta?
        //bippr epsilon? delta?
        double raw_epsilon = config.epsilon, ratio = 1;
        fora_bippr_setting(graph.n, graph.m, ratio, raw_epsilon);
        display_setting();
        used_counter = FB_QUERY;
        //使用vector代替map，提前预设向量大小为节点大小
        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);
        bwd_idx.first.initialize(graph.n);
        bwd_idx.second.initialize(graph.n);

        rw_counter.initialize(graph.n);
        for (int i = 0; i < query_size; i++) {
            cout << i + 1 << ". source node:" << queries[i] << endl;
            fora_bippr_query(queries[i], graph, raw_epsilon);
            INFO(Timer::used(5));
            INFO(Timer::used(6));
            INFO(Timer::used(9));
            split_line();
        }
        cout << "num_total_fo" << num_total_fo << endl << "num_total_bi" << num_total_bi << endl;
    } else if (config.algo == FORA) { //fora
        fora_setting(graph.n, graph.m);
        display_setting();
        used_counter = FORA_QUERY;
        //使用vector代替map，提前预设向量大小为节点大小
        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);

        // if(config.multithread)
        //     vec_ppr.resize(graph.n);

        // rw_counter.initialize(graph.n);
        for (int i = 0; i < query_size; i++) {
            cout << i + 1 << ". source node:" << queries[i] << endl;
            fora_query_basic(queries[i], graph);
            split_line();
        }
    }
    display_time_usage(used_counter, query_size);
    set_result(graph, used_counter, query_size);
}

void query(Graph &graph) {
    INFO(config.algo);
    vector<int> queries;
    load_ss_query(queries);
    unsigned int query_size = queries.size();
    query_size = min(query_size, config.query_size);
    INFO(query_size);
    int used_counter = 0;

    // assert(config.rw_cost_ratio >= 0);
    // INFO(config.rw_cost_ratio);

    assert(config.rmax_scale >= 0);
    INFO(config.rmax_scale);

    // ppr.initialize(graph.n);
    ppr.init_keys(graph.n);
    // sfmt_init_gen_rand(&sfmtSeed , 95082);
    if (config.algo == MC_DHT) {
        montecarlo_setting();
        display_setting();
        used_counter = MC_QUERY;

        rw_counter.initialize(graph.n);

        for (int i = 0; i < query_size; i++) {
            cout << i + 1 << ". source node:" << queries[i] << endl;
            montecarlo_query_dht(queries[i], graph);
            split_line();
        }
    } else if (config.algo == BIPPR) { //bippr
        bippr_setting(graph.n, graph.m);
        display_setting();
        used_counter = BIPPR_QUERY;

        bwd_idx.first.initialize(graph.n);
        bwd_idx.second.initialize(graph.n);

        rw_counter.initialize(graph.n);
        for (int i = 0; i < query_size; i++) {
            cout << i + 1 << ". source node:" << queries[i] << endl;
            bippr_query(queries[i], graph);
            split_line();
        }
    } else if (config.algo == HUBPPR) {
        bippr_setting(graph.n, graph.m);
        display_setting();
        used_counter = HUBPPR_QUERY;

        bwd_idx.first.initialize(graph.n);
        bwd_idx.second.initialize(graph.n);
        hub_counter.initialize(graph.n);
        rw_counter.initialize(graph.n);

        load_hubppr_oracle(graph);
        for (int i = 0; i < query_size; i++) {
            cout << i + 1 << ". source node:" << queries[i] << endl;
            hubppr_query(queries[i], graph);
            split_line();
        }
    } else if (config.algo == FORA) { //fora
        fora_setting(graph.n, graph.m);
        display_setting();
        used_counter = FORA_QUERY;
        //使用vector代替map，提前预设向量大小为节点大小
        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);

        // if(config.multithread)
        //     vec_ppr.resize(graph.n);

        // rw_counter.initialize(graph.n);
        for (int i = 0; i < query_size; i++) {
            cout << i + 1 << ". source node:" << queries[i] << endl;
            fora_query_basic(queries[i], graph);
            split_line();
        }
    } else if (config.algo == FORA_MC) {
        //先按照FORA的流程走一遍，然后每一个查询，结束时，进行一遍蒙特卡洛
        fora_setting(graph.n, graph.m);
        display_setting();
        used_counter = FORA_MC_QUERY;

        Timer timer(used_counter);
        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);
        rw_counter.initialize(graph.n);

        // if(config.multithread)
        //     vec_ppr.resize(graph.n);

        // rw_counter.initialize(graph.n);
        for (int i = 0; i < query_size; i++) {
            cout << i + 1 << ". source node:" << queries[i] << endl;
            fora_query_basic(queries[i], graph);
            int tmp = 0;
            for (int i = 0; i < graph.n; i++) {
                if (graph.g[i].size() > 0) {//判断这个点是否存在
                    if (fwd_idx.second.notexist(i) || fwd_idx.second[i] == 0)//residue为零或者不存在的进行蒙特卡洛
                    {
                        tmp++;
                        //montecarlo_query2(i+1, graph);
                    }
                }
            }
            cout << tmp << endl;
            split_line();
        }

    } else if (config.algo == MC) { //mc
        montecarlo_setting();
        display_setting();
        used_counter = MC_QUERY;

        rw_counter.initialize(graph.n);

        for (int i = 0; i < query_size; i++) {
            cout << i + 1 << ". source node:" << queries[i] << endl;
            montecarlo_query(queries[i], graph);
            split_line();
        }
    } else if (config.algo == MC2) { //mc
        montecarlo_setting2();
        display_setting();
        used_counter = MC_QUERY;

        rw_counter.initialize(graph.n);

        for (int i = 0; i < graph.n; i++) {
            if (graph.g[i].size() > 0) {//判断这个点是否存在
                montecarlo_query2(i, graph);
            }
        }
    } else if (config.algo == FWDPUSH) {
        fwdpush_setting(graph.n, graph.m);
        display_setting();
        used_counter = FWD_LU;

        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);

        for (int i = 0; i < query_size; i++) {
            cout << i + 1 << ". source node:" << queries[i] << endl;
            Timer timer(used_counter);
            double rsum = 1;
            forward_local_update_linear(queries[i], graph, rsum, config.rmax);
            compute_ppr_with_reserve();
            split_line();
        }
    }

    display_time_usage(used_counter, query_size);
    set_result(graph, used_counter, query_size);
}


void batch_topk(Graph &graph) {
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

    // not FORA, so it's of single source
    // no need to change k to run again
    // check top-k results for different k
    if (config.algo != FORA && config.algo != HUBPPR) {
        for (int i = 0; i < query_size; i++) {
            cout << i + 1 << ". source node:" << queries[i] << endl;
            get_topk(queries[i], graph);
            split_line();
        }

        display_time_usage(used_counter, query_size);
        set_result(graph, used_counter, query_size);

        display_precision_for_dif_k();
    } else { // for FORA, when k is changed, run algo again
        for (unsigned int k: ks) {
            config.k = k;
            INFO("========================================");
            INFO("k is set to be ", config.k);
            result.topk_recall = 0;
            result.topk_precision = 0;
            result.real_topk_source_count = 0;
            Timer::clearAll();
            for (int i = 0; i < query_size; i++) {
                cout << i + 1 << ". source node:" << queries[i] << endl;
                get_topk(queries[i], graph);
                split_line();
            }
            pred_results[k].topk_precision = result.topk_precision;
            pred_results[k].topk_recall = result.topk_recall;
            pred_results[k].real_topk_source_count = result.real_topk_source_count;

            cout << "k=" << k << " precision=" << result.topk_precision / result.real_topk_source_count
                 << " recall=" << result.topk_recall / result.real_topk_source_count << endl;
            cout << "Average query time (s):" << Timer::used(used_counter) / query_size << endl;
            Timer::reset(used_counter);
        }

        // display_time_usage(used_counter, query_size);
        display_precision_for_dif_k();
    }
}

void batch_topk_dht(Graph &graph) {
    vector<int> queries;
    load_ss_query(queries);
    INFO(queries.size());
    unsigned int query_size = queries.size();
    query_size = min(query_size, config.query_size);
    vector<int> true_queries;
    int used_counter = 0;

    assert(config.k < graph.n - 1);
    assert(config.k > 1);
    INFO(config.k);

    split_line();

    load_exact_topk_ppr();
    map<int, double> ppr_self = load_exact_self_ppr();
    //INFO(ppr_self[62505]);

    bwd_idx.first.initialize(graph.n);
    bwd_idx.second.initialize(graph.n);
    //config.rmax=0.0001;
    int cur = 0;
    for (auto item:exact_topk_pprs) {
        if (++cur > config.query_size)break;
        int source_node = item.first;
        true_queries.emplace_back(source_node);
        //INFO(source_node);
        unordered_map<int, double> dht;
        vector<pair<int, double>> topk_pprs = item.second;
        for (int k = 0; k < topk_pprs.size(); ++k) {
            int node = topk_pprs[k].first;

            if (ppr_self.find(node) != ppr_self.end()) {
                dht[node] = topk_pprs[k].second / ppr_self.at(node);
                //INFO(dht[node]);
            } else {
                //INFO(node);
                //reverse_local_update_linear(node,graph);
                //INFO(topk_pprs[k].second / bwd_idx.first[node]);
            }
        }
        vector<pair<int, double>> temp_top_dht(config.k);
        partial_sort_copy(dht.begin(), dht.end(), temp_top_dht.begin(), temp_top_dht.end(),
                          [](pair<int, double> const &l, pair<int, double> const &r) { return l.second > r.second; });
        exact_topk_dhts[source_node] = temp_top_dht;
        //INFO(temp_top_dht);
    }

    // not FORA, so it's single source
    // no need to change k to run again
    // check top-k results for different k
    if (config.algo == MC_DHT) {
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
    if (config.algo == MC_DHT) {
        rw_counter.initialize(graph.n);
        dht.initialize(graph.n);
        montecarlo_setting();
    } else if (config.algo == DNE) {
        dht.initialize(graph.n);
    } else if (config.algo == FB_RAW) {
        //先使用FORA，然后使用1/alpha ~ 1这个范围，计算这些点的BiPPR
        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);
        rw_counter.init_keys(graph.n);
        upper_bounds.init_keys(graph.n);
        lower_bounds.init_keys(graph.n);
        ppr.initialize(graph.n);
        //使用vector代替map，提前预设向量大小为节点大小
        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);
        bwd_idx.first.initialize(graph.n);
        bwd_idx.second.initialize(graph.n);
        double raw_epsilon = config.epsilon, ratio = 1;
        fb_raw_setting(graph.n, graph.m, ratio, raw_epsilon);
    } else if (config.algo == FB) {
        //先利用节点的度得到初步的范围，
        // 另外重复利用RW？
        // 然后维护真实TOPK，计算UB的nth-k，对topk LB进行过滤，在候选集比较小的时候，使用哈希表保存，另外比较成本，使用成本低的。
        fwd_idx.first.initialize(graph.n);//forward p
        fwd_idx.second.initialize(graph.n);//forward r
        rw_counter.init_keys(graph.n);//
        rw_bippr_counter.init_keys(graph.n);
        upper_bounds.init_keys(graph.n);
        lower_bounds.init_keys(graph.n);
        upper_bounds_self.init_keys(graph.n);
        upper_bounds_self_init.init_keys(graph.n);
        lower_bounds_self.init_keys(graph.n);
        upper_bounds_dht.init_keys(graph.n);
        lower_bounds_dht.init_keys(graph.n);
        ppr.initialize(graph.n);
        ppr_bi.initialize(graph.n);
        dht.initialize(graph.n);
        topk_filter.initialize(graph.n);
    }
    if (config.algo == MC_DHT || config.algo == DNE) {
        for (int i = 0; i < query_size; i++) {
            cout << i + 1 << ". source node:" << true_queries[i] << endl;
            get_topk2(true_queries[i], graph);
            if (config.algo == MC_DHT && (graph.data_folder.find("lj"))) {
                compute_NDCG_for_dif_k_dht(true_queries[i], ppr_self, graph);
            }
            split_line();
        }
    }
    for (int i = 0; i < query_size; i++) {
        cout << i + 1 << ". source node:" << true_queries[i] << endl;
        get_topk2(true_queries[i], graph);
        if (config.algo == MC_DHT) {
            compute_NDCG_for_dif_k_dht(true_queries[i], ppr_self, graph);
        }
        compute_NDCG(true_queries[i], ppr_self, graph);
        split_line();
    }

    cout << "average iter times:" << num_iter_topk / query_size << endl;
    display_time_usage(used_counter, query_size);
    set_result(graph, used_counter, query_size);

    //not FORA, so it's single source
    //no need to change k to run again
    // check top-k results for different k
    if (config.algo == MC_DHT) {
        display_precision_for_dif_k();
    }
}

#endif //FORA_QUERY_H
