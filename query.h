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
    int dne_m = ceil(pow(deg_graph, log(graph.n) / log(deg_graph / (1 - config.alpha))));
    for (int l = 0; l < dht.m_num; ++l) {
        dht.insert(l, dhe_query_basic(v, l, dne_m, graph));
    }
#ifdef CHECK_PPR_VALUES
    display_dht();
#endif
}

void montecarlo_query2(int v, const Graph &graph) {
    Timer timer(MC_QUERY2);

    double fwd_rw_count = 3 * log(2 / config.pfail) / config.epsilon / config.epsilon / config.alpha;
    rw_counter.clean();
    ppr.reset_zero_values();

    {
        Timer tm(RONDOM_WALK2);
        num_total_rw += fwd_rw_count;
        for (unsigned long i = 0; i < fwd_rw_count; i++) {
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
        ppr[node_id] = rw_counter[node_id] * 1.0 / fwd_rw_count;
    }

#ifdef CHECK_PPR_VALUES
    display_ppr();
#endif
}

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

void bippr_query_with_fora(const Graph &graph, double check_rsum) {
    ppr.reset_zero_values();
    ppr_bi.initialize(graph.n);
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
    for (int k = 0; k < ppr.m_num; ++k) {
        if (ppr[k] <= 0 && (fwd_idx.second.notexist(k) || fwd_idx.second[k] <= 0))continue;
        counter++;
        {
            Timer tm(RONDOM_WALK);
            double residual = fwd_idx.second[k] > 0 ? fwd_idx.second[k] : 0;
            assert(residual < config.rmax);
            assert(residual < config2.rmax);
            unsigned long num_s_rw = ceil(residual / check_rsum * num_random_walk);
            num_s_rw = ceil(config2.omega) > num_s_rw ? ceil(config2.omega) : num_s_rw;
            double a_s = residual / check_rsum * num_random_walk / num_s_rw;

            double ppr_incre = a_s * check_rsum / num_random_walk;
            num_total_rw += num_s_rw;
            for (unsigned long i = 0; i < num_s_rw; i++) {
                int destination = random_walk(k, graph);
                ppr[destination] += ppr_incre;
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
                if (rw_counter.occur[node_id] == k) {
                    ppr_bi[k] += rw_counter[node_id] * 1.0 / config2.omega * residual;
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

void hubppr_query(int s, const Graph &graph) {
    Timer timer(HUBPPR_QUERY);

    ppr.reset_zero_values();

    {
        Timer tm(RONDOM_WALK);
        fwd_with_hub_oracle(graph, s);
        count_hub_dest();
        INFO("finish fwd work", hub_counter.occur.m_num, rw_counter.occur.m_num);
    }

    {
        Timer tm(BWD_LU);
        for (int t = 0; t < graph.n; t++) {
            bwd_with_hub_oracle(graph, t);
            // reverse_local_update_linear(t, graph);
            if ((bwd_idx.first.notexist(s) || 0 == bwd_idx.first[s]) && 0 == bwd_idx.second.occur.m_num) {
                continue;
            }

            if (rw_counter.occur.m_num < bwd_idx.second.occur.m_num) { //iterate on smaller-size list
                for (int i = 0; i < rw_counter.occur.m_num; i++) {
                    int node = rw_counter.occur[i];
                    if (bwd_idx.second.exist(node)) {
                        ppr[t] += bwd_idx.second[node] * rw_counter[node];
                    }
                }
            } else {
                for (int i = 0; i < bwd_idx.second.occur.m_num; i++) {
                    int node = bwd_idx.second.occur[i];
                    if (rw_counter.exist(node)) {
                        ppr[t] += rw_counter[node] * bwd_idx.second[node];
                    }
                }
            }
            ppr[t] = ppr[t] / config.omega;
            if (bwd_idx.first.exist(s))
                ppr[t] += bwd_idx.first[s];
        }
    }

#ifdef CHECK_PPR_VALUES
    display_ppr();
#endif
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

    unsigned long long num_random_walk = config.omega * check_rsum;//这里的omega并不是真正的Omega，num_random_walk才是
    INFO(num_random_walk);
    // INFO(num_random_walk);
    //num_total_rw += num_random_walk;

    {
        Timer timer(RONDOM_WALK); //both rand-walk and source distribution are related with num_random_walk
        //Timer tm(SOURCE_DIST);
        if (config.with_rw_idx) {
            fwd_idx.second.occur.Sort();
            for (long i = 0; i < fwd_idx.second.occur.m_num; i++) {
                int source = fwd_idx.second.occur[i];
                double residual = fwd_idx.second[source];
                unsigned long num_s_rw = ceil(residual / check_rsum * num_random_walk);
                double a_s = residual / check_rsum * num_random_walk / num_s_rw;

                double ppr_incre = a_s * check_rsum / num_random_walk;

                num_total_rw += num_s_rw;

                //for each source node, get rand walk destinations from previously generated idx or online rand walks
                if (num_s_rw >
                    rw_idx_info[source].second) { //if we need more destinations than that in idx, rand walk online
                    for (unsigned long k = 0; k < rw_idx_info[source].second; k++) {
                        int des = rw_idx[rw_idx_info[source].first + k];
                        ppr[des] += ppr_incre;
                    }
                    num_hit_idx += rw_idx_info[source].second;

                    for (unsigned long j = 0; j < num_s_rw - rw_idx_info[source].second; j++) { //rand walk online
                        int des = random_walk(source, graph);
                        ppr[des] += ppr_incre;
                    }
                } else { // using previously generated idx is enough
                    for (unsigned long k = 0; k < num_s_rw; k++) {
                        int des = rw_idx[rw_idx_info[source].first + k];
                        ppr[des] += ppr_incre;
                    }
                    num_hit_idx += num_s_rw;
                }
            }
        } else { //rand walk online
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
}

void compute_ppr_with_fwdidx_topk_with_bound(const Graph &graph, double check_rsum) {
    compute_ppr_with_reserve();

    if (check_rsum == 0.0)
        return;

    long num_random_walk = config.omega * check_rsum;
    long real_num_rand_walk = 0;

    {
        Timer timer(RONDOM_WALK); //both rand-walk and source distribution are related with num_random_walk

        //Timer tm(SOURCE_DIST);
        if (config.with_rw_idx) { //rand walk with previously generated idx
            fwd_idx.second.occur.Sort();
            //for each source node, get rand walk destinations from previously generated idx or online rand walks
            for (long i = 0; i < fwd_idx.second.occur.m_num; i++) {
                int source = fwd_idx.second.occur[i];
                double residual = fwd_idx.second[source];
                long num_s_rw = ceil(residual / check_rsum * num_random_walk);
                double a_s = residual / check_rsum * num_random_walk / num_s_rw;

                double ppr_incre = a_s * check_rsum / num_random_walk;

                num_total_rw += num_s_rw;
                real_num_rand_walk += num_s_rw;

                long num_used_idx = 0;
                bool source_cnt_exist = rw_counter.exist(source);
                if (source_cnt_exist)
                    num_used_idx = rw_counter[source];
                long num_remaining_idx = rw_idx_info[source].second - num_used_idx;

                if (num_s_rw <= num_remaining_idx) {
                    // using previously generated idx is enough
                    long k = 0;
                    for (; k < num_remaining_idx; k++) {
                        if (k < num_s_rw) {
                            int des = rw_idx[rw_idx_info[source].first + k];
                            if (ppr.exist(des))
                                ppr[des] += ppr_incre;
                            else
                                ppr.insert(des, ppr_incre);
                        } else
                            break;
                    }

                    if (source_cnt_exist) {
                        rw_counter[source] += k;
                    } else {
                        rw_counter.insert(source, k);
                    }

                    num_hit_idx += k;
                } else {
                    //we need more destinations than that in idx, rand walk online
                    for (long k = 0; k < num_remaining_idx; k++) {
                        int des = rw_idx[rw_idx_info[source].first + k];
                        if (!ppr.exist(des))
                            ppr.insert(des, ppr_incre);
                        else
                            ppr[des] += ppr_incre;
                    }
                    num_hit_idx += num_remaining_idx;

                    if (!source_cnt_exist) {
                        rw_counter.insert(source, num_remaining_idx);
                    } else {
                        rw_counter[source] += num_remaining_idx;
                    }

                    for (long j = 0; j < num_s_rw - num_remaining_idx; j++) { //rand walk online
                        int des = random_walk(source, graph);
                        if (!ppr.exist(des))
                            ppr.insert(des, ppr_incre);
                        else
                            ppr[des] += ppr_incre;
                    }
                }
            }
        } else { //rand walk online
            for (long i = 0; i < fwd_idx.second.occur.m_num; i++) {
                int source = fwd_idx.second.occur[i];
                double residual = fwd_idx.second[source];
                long num_s_rw = ceil(residual / check_rsum * num_random_walk);
                double a_s = residual / check_rsum * num_random_walk / num_s_rw;

                real_num_rand_walk += num_s_rw;

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

    set_ppr_bounds(graph, check_rsum, real_num_rand_walk);
}

void fora_bippr_query(int v, const Graph &graph, double raw_epsilon) {
    Timer timer(FB_QUERY);
    double rsum = 1.0, ratio = 1;//sqrt(config.alpha);
    static vector<int> forward_from;
    forward_from.clear();
    forward_from.reserve(graph.n);
    forward_from.push_back(v);
    fwd_idx.first.clean();  //reserve
    fwd_idx.second.clean();  //residual
    fwd_idx.second.insert(v, rsum);
    //fora_bippr_setting(graph.n, graph.m, 1,raw_epsilon);
    {
        Timer timer(FORA_QUERY);
        //初始时，设置两者一样，然后第一次检测大小关系，
        {
            Timer timer(FWD_LU);

            do {
                cout << endl;
                INFO(ratio);
                //display_setting();
                fora_bippr_setting(graph.n, graph.m, ratio, raw_epsilon);
                cout << "epsilon:\t" << config.epsilon << "\t" << config2.epsilon << endl;
                cout << "omega:\t" << config.omega << "\t" << config2.omega << endl;
                cout << (1 + config.epsilon) / (1 - config2.epsilon) << endl;
                forward_local_update_linear_topk_dht(v, graph, rsum, config.rmax, 0, forward_from);
            } while (false);
            (check_cost(rsum, ratio, graph.n, graph.m));

            //forward_local_update_linear(v, graph, rsum, config.rmax); //forward propagation, obtain reserve and residual
        }
        // compute_ppr_with_fwdidx(graph);
        bippr_query_with_fora(graph, rsum);
    }
    INFO(config2.omega * fwd_idx.second.occur.m_num);
    int num_rw_bi = num_total_rw;
    INFO(num_rw_bi);
    INFO(graph.m * config.rmax * config.omega);
    INFO(rsum * config.omega);
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

    while (config.delta >= min_delta) {
        fora_setting(graph.n, graph.m);
        num_iter_topk++;

        {
            Timer timer(FWD_LU);
            forward_local_update_linear_topk(v, graph, rsum, config.rmax, lowest_delta_rmax,
                                             forward_from); //forward propagation, obtain reserve and residual
        }

        compute_ppr_with_fwdidx_topk_with_bound(graph, rsum);
        if (if_stop() || config.delta <= min_delta) {
            break;
        } else
            config.delta = max(min_delta, config.delta / 2.0);  // otherwise, reduce delta to delta/2
    }
}


iMap<int> updated_pprs;

void hubppr_query_topk_martingale(int s, const Graph &graph) {
    unsigned long long the_omega =
            2 * config.rmax * log(2 * config.k / config.pfail) / config.epsilon / config.epsilon / config.delta;
    static double bwd_cost_div = 1.0 * graph.m / graph.n / config.alpha;

    static double min_ppr = 1.0 / graph.n;
    static double new_pfail = config.pfail / 2.0 / graph.n / log2(1.0 * graph.n * config.alpha * graph.n * graph.n);
    static double pfail_star = log(new_pfail / 2);

    static std::vector<bool> target_flag(graph.n);
    static std::vector<double> m_omega(graph.n);
    static vector<vector<int>> node_targets(graph.n);
    static double cur_rmax = 1;

    // rw_counter.clean();
    for (int t = 0; t < graph.n; t++) {
        map_lower_bounds[t].second = 0;//min_ppr;
        upper_bounds[t] = 1.0;
        target_flag[t] = true;
        m_omega[t] = 0;
    }

    int num_iter = 1;
    int target_size = graph.n;
    if (cur_rmax > config.rmax) {
        cur_rmax = config.rmax;
        for (int t = 0; t < graph.n; t++) {
            if (target_flag[t] == false)
                continue;
            reverse_local_update_topk(s, t, reserve_maps[t], cur_rmax, residual_maps[t], graph);
            for (const auto &p: residual_maps[t]) {
                node_targets[p.first].push_back(t);
            }
        }
    }
    while (target_size > config.k &&
           num_iter <= 64) { //2^num_iter <= 2^64 since 2^64 is the largest unsigned integer here
        unsigned long long num_rw = pow(2, num_iter);
        rw_counter.clean();
        generate_accumulated_fwd_randwalk(s, graph, num_rw);
        updated_pprs.clean();
        // update m_omega
        {
            for (int x = 0; x < rw_counter.occur.m_num; x++) {
                int node = rw_counter.occur[x];
                for (const int t: node_targets[node]) {
                    if (target_flag[t] == false)
                        continue;
                    m_omega[t] += rw_counter[node] * residual_maps[t][node];
                    if (!updated_pprs.exist(t))
                        updated_pprs.insert(t, 1);
                }
            }
        }

        double b = (2 * num_rw - 1) * pow(cur_rmax / 2.0, 2);
        double lambda = sqrt(pow(cur_rmax * pfail_star / 3, 2) - 2 * b * pfail_star) - cur_rmax * pfail_star / 3;
        {
            for (int i = 0; i < updated_pprs.occur.m_num; i++) {
                int t = updated_pprs.occur[i];
                if (target_flag[t] == false)
                    continue;

                double reserve = 0;
                if (reserve_maps[t].find(s) != reserve_maps[t].end()) {
                    reserve = reserve_maps[t][s];
                }
                set_martingale_bound(lambda, 2 * num_rw - 1, t, reserve, cur_rmax, pfail_star, min_ppr, m_omega[t]);
            }
        }

        topk_pprs.clear();
        topk_pprs.resize(config.k);
        partial_sort_copy(map_lower_bounds.begin(), map_lower_bounds.end(), topk_pprs.begin(), topk_pprs.end(),
                          [](pair<int, double> const &l, pair<int, double> const &r) { return l.second > r.second; });

        double k_bound = topk_pprs[config.k - 1].second;
        if (k_bound * (1 + config.epsilon) >= upper_bounds[topk_pprs[config.k - 1].first] ||
            (num_rw >= the_omega && cur_rmax <= config.rmax)) {
            break;
        }

        for (int t = 0; t < graph.n; t++) {
            if (target_flag[t] == true && upper_bounds[t] <= k_bound) {
                target_flag[t] = false;
                target_size--;
            }
        }
        num_iter++;
    }
}

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

    Timer timer(0);
    const static double min_delta = 1.0 / graph.n;
    const static double init_delta = 1.0 / 4;
    const static double min_epsilon = config.epsilon / (graph.n / config.k + 1 + config.epsilon);
    const static double init_epsilon = config.epsilon / 2;
    threshold = (1.0 - config.ppr_decay_alpha) / pow(500, config.ppr_decay_alpha) /
                pow(graph.n, 1 - config.ppr_decay_alpha);//?
    //////////应该对两个的epsilon进行设置
    config.epsilon = config.epsilon / (1 + (1 + config.epsilon) * config.k / graph.n);
    const static double new_pfail = 1.0 / graph.n / graph.n / log(graph.n);

    config.pfail = new_pfail;  // log(1/pfail) -> log(1*n/pfail)
    config.pfail /= 2;
    config2 = config;
    config.delta = init_delta;
    config2.delta = config2.alpha;
    config2.epsilon = init_epsilon;

    const static double lowest_delta_rmax =
            config.epsilon * sqrt(min_delta / 3 / graph.m / log(1 / new_pfail));//更新后的r_max
    const static double lowest_delta_rmax_2 =
            min_epsilon * sqrt(graph.m * 1.0 * config2.delta / graph.n / 3 / log(1 / new_pfail));

    double rsum = 1.0;

    static vector<int> forward_from;
    forward_from.clear();
    forward_from.reserve(graph.n);
    forward_from.push_back(v);
    static unordered_map<int, vector<int>> backward_from;

    fwd_idx.first.clean();  //reserve
    fwd_idx.second.clean();  //residual
    fwd_idx.second.insert(v, rsum);
    multi_bwd_idx_p.clear();
    multi_bwd_idx_r.clear();
    multi_bwd_idx_rw.clear();

    zero_ppr_upper_bound = 1.0;

    if (config.with_rw_idx)
        rw_counter.reset_zero_values(); //store the pointers of each node's used rand-walk idxs

    // for delta: try value from 1/4 to 1/n
    int iteration = 0;
    upper_bounds.reset_one_values();
    lower_bounds.reset_zero_values();
    upper_bounds_dht.reset_values(1 / config.alpha);
    lower_bounds_dht.reset_zero_values();
    lower_bounds_self.reset_values(config.alpha);
    // 初始化上下界
    for (int j = 0; j < graph.n; ++j) {
        int degree_re = min(graph.g[j].size(), graph.gr[j].size());
        int degree_all = 0;
        for (int nei:graph.gr[j]) {
            degree_all += graph.gr[nei].size();
        }
        if (degree_all == 0) {
            upper_bounds_self.insert(j, 1);
            upper_bounds_self_init.insert(j, 1);
        } else {
            upper_bounds_self.insert(j, config.alpha +
                                        (1 - config.alpha) * (1 - config.alpha) * config.alpha * degree_re /
                                        degree_all +
                                        (1 - config.alpha) * (1 - config.alpha) * (1 - config.alpha) * config.alpha *
                                        (1 - degree_re * 1.0 / degree_all));
            upper_bounds_self_init.insert(j, upper_bounds_self[j]);
        }
    }
    bool fora_flag = true, stop, first = true;
///考虑设置一个值略过前面一些轮
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

    topk_dht();

    // not FORA, so it's single source 如果不是FORA或者HUBPPR就不用重复进行实验，直接计算不同k下的准确率
    // no need to change k to run again
    // check top-k results for different k
    if (config.algo != FORA && config.algo != HUBPPR) {
        compute_precision_for_dif_k(v);
    }

    //compute_precision(v);//如果有准确的值的话，就计算准确率

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
    double myeps = 1.0 / graph.n;
    q[pointer_q++] = start;
    bwd_idx_th.second.occur[start] = start;
    bwd_idx_th.second[start] = 1;

    idx[start] = start;
    while (left != pointer_q) {
        int v = q[left++];
        left%=graph.n;
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
                    pointer_q%=graph.n;
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
    }
}

void multi_power_iter_self(const Graph &graph, const vector<int> &source,
                           unordered_map<int, double> &map_self_ppr) {
    //static thread_local unordered_map<int, double> map_ppr;
    static thread_local Bwdidx bwd_idx_th;
    bwd_idx_th.first.initialize(graph.n);
    bwd_idx_th.second.initialize(graph.n);
    fill(bwd_idx_th.first.occur.m_data, bwd_idx_th.first.occur.m_data + graph.n, -1);
    fill(bwd_idx_th.second.occur.m_data, bwd_idx_th.second.occur.m_data + graph.n, -1);
    static thread_local vector<int> idx(graph.n, -1), q(graph.n);
    for (int start: source) {
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


    split_line();
    // montecarlo_setting();

    unsigned NUM_CORES = std::thread::hardware_concurrency() - 2;
    assert(NUM_CORES >= 2);

    int num_thread = NUM_CORES;
    int avg_queries_per_thread = graph.n / num_thread;

    vector<vector<int>> source_for_all_core(num_thread);
    vector<unordered_map<int, double >> ppv_self_for_all_core(num_thread);

    for (int tid = 0; tid < num_thread; tid++) {
        int s = tid * avg_queries_per_thread;
        int t = s + avg_queries_per_thread;

        if (tid == num_thread - 1)
            t += graph.n % num_thread;

        for (; s < t; s++) {
            // cout << s+1 <<". source node:" << queries[s] << endl;
            source_for_all_core[tid].push_back(s);
        }
    }


    {
        Timer timer(PI_QUERY);
        INFO("power itrating...");
        std::vector<std::future<void> > futures(num_thread);
        for (int tid = 0; tid < num_thread; tid++) {
            futures[tid] = std::async(std::launch::async, multi_power_iter_self, std::ref(graph),
                                      std::ref(source_for_all_core[tid]), std::ref(ppv_self_for_all_core[tid]));
        }
        std::for_each(futures.begin(), futures.end(), std::mem_fn(&std::future<void>::wait));
    }

    // cout << "average iter times:" << num_iter_topk/query_size << endl;
    cout << "average generation time (s): " << Timer::used(PI_QUERY) * 1.0 / graph.n << endl;

    INFO("combine results...");
    vector<double> ppr_self(graph.n);
    for (int tid = 0; tid < num_thread; tid++) {
        for (auto &ppv: ppv_self_for_all_core[tid]) {
            //exact_topk_pprs.insert(ppv);
            ppr_self[ppv.first] = ppv.second;
        }
    }
    save_self_ppr(ppr_self);
    //save_exact_topk_ppr();
}

void gen_exact_topk(const Graph &graph) {
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
    for (int tid = 0; tid < num_thread; tid++) {
        for (auto &ppv: ppv_for_all_core[tid]) {
            exact_topk_pprs.insert(ppv);
        }
        ppv_for_all_core[tid].clear();
    }

    save_exact_topk_ppr();
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
    if (config.algo != MC_DHT) {
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
        //bippr_setting(graph.n, graph.m); 设置rmax 和omega

        for (int i = 0; i < query_size; i++) {
            cout << i + 1 << ". source node:" << queries[i] << endl;
            get_topk2(queries[i], graph);
            split_line();
        }
    } else {

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
            split_line();
        }
    } else if (config.algo == FB) {
        //fora epsilon? delta?
        //bippr epsilon? delta?
        double raw_epsilon = config.epsilon, ratio = 1;
        config.pfail /= 2;
        config.delta *= config.alpha;
        config2 = config;
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

#endif //FORA_QUERY_H
