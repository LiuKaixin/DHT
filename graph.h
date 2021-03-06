#ifndef __GRAPH_H__
#define __GRAPH_H__

#include "mylib.h"
#include "config.h"

class Graph {
public:
    //两个向量以起点和终点两种方式来存储边
    vector<vector<int>> g;
    vector<vector<int>> gr;
    string data_folder;

    //vector<double> global_ppr;

    // node rank[100] = 0, means node 100 has first rank
    vector<int> node_rank;
    // node_score[0]
    vector<double> node_score;

    //node order 0 = [100, 34.5], most important node is 100 with score 34.5
    vector<pair<int, double>> node_order;
    vector<int> loc;


    // the tele ratio for random walk
    double alpha;

    static bool cmp(const pair<int, double> &t1, const pair<int, double> &t2) {
        return t1.second > t2.second;
    }

    int n;
    long long m;

    Graph(string data_folder) {
        INFO("sub constructor");
        this->data_folder = data_folder;
        this->alpha = ALPHA_DEFAULT;
        if(config.action == GEN_SS_QUERY && !config.query_high_degree)
            init_nm();
        else
            init_graph();
        cout << "init graph n: " << this->n << " m: " << this->m << endl;
    }

    //读取属性文件中的n和m的值
    void init_nm() {
        string attribute_file = data_folder + FILESEP + "attribute.txt";
        assert_file_exist("attribute file", attribute_file);
        ifstream attr(attribute_file);
        string line1, line2;
        char c;
        while (true) {
            attr >> c;
            if (c == '=') break;
        }
        attr >> n;
        while (true) {
            attr >> c;
            if (c == '=') break;
        }
        attr >> m;
    }

    void init_graph() {
        init_nm();
        g = vector<vector<int>>(n, vector<int>());
        gr = vector<vector<int>>(n, vector<int>());
        string graph_file = data_folder + FILESEP + "graph.txt";
        assert_file_exist("graph file", graph_file);
        FILE *fin = fopen(graph_file.c_str(), "r");
        int t1, t2;
        if (data_folder.find("dblp2010")!=string::npos||data_folder.find("orkut")!=string::npos){
            while (fscanf(fin, "%d%d", &t1, &t2) != EOF) {
                assert(t1 < n);
                assert(t2 < n);
                if(t1 == t2) continue;
                g[t1].push_back(t2);
                gr[t2].push_back(t1);
                g[t2].push_back(t1);
                gr[t1].push_back(t2);
            }
            m*=2;
        } else {
            while (fscanf(fin, "%d%d", &t1, &t2) != EOF) {
                assert(t1 < n);
                assert(t2 < n);
                if(t1 == t2) continue;
                g[t1].push_back(t2);
                gr[t2].push_back(t1);
            }
        }

    }

    void dfs_cycle(int u, int p, int color[], int par[], int &cyclenumber, int cycles[], ofstream &file) {

        Timer timer(DFS_CYCLE);
        // already (completely) visited vertex.
        if (color[u] == 2) {
            return;
        }

        // seen vertex, but was not completely visited -> cycle detected.
        // backtrack based on parents to find the complete cycle.
        if (color[u] == 1) {

            cyclenumber++;
            if (cyclenumber % 2000 == 0) {
                cout << "cycle Number =" << cyclenumber << endl;
            }
            /*
            int cur = p;
            //file << "Cycle number" << cyclenumber << ": " << cur;
            int cycle_lenth=1;
            // backtrack the vertex which are
            // in the current cycle thats found
            while (cur != u) {
                cycle_lenth++;
                cur = par[cur];
                //file << " " << cur;
            }
            //file << "\n";
            cycles[cycle_lenth]++;
            */
            return;
        }
        par[u] = p;

        // partially visited.
        color[u] = 1;

        // simple dfs on graph
        for (int v : g[u]) {

            // if it has not been visited previously
            if (v == par[u]) {
                continue;
            }
            dfs_cycle(v, u, color, par, cyclenumber, cycles, file);
        }

        // completely visited.
        color[u] = 2;
    }

    double get_avg_degree() const {
        return double(m) / double(n);
    }


};



static void init_parameter(Config &config, const Graph &graph) {
    // init the bwd delta, fwd delta etc

    INFO("init parameters", graph.n);
    config.delta = 1.0 / graph.n;
    config.pfail = 1.0 / graph.n;

    config.dbar = double(graph.m) / double(graph.n); // average degree


}



#endif
