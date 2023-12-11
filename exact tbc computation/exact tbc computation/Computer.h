#pragma once
#include"Transformer.h"
#include <stack>
#include <deque> 
#include <set>
#include <queue>

using namespace std;
// set thread num
#define THREAD_NUM 1
struct  Computer
{
	Transformer transformer;
	vector<double> BC;
	TransformGraph& Tran_Graph = transformer.tran_graph;
	CompressGraph& Cmp_Graph = transformer.cmp_graph;
	unordered_map<string, int>& id_to_seq = transformer.data_graph.identity_to_seq;
	int data_nodes_size = transformer.data_graph.graph_data.size();
	unordered_map<int, int>& tran_root_seqs = transformer.tran_root_seqs;
	Computer(Transformer& t) :
		transformer(t) {}
	//compute shortest path and temporal betweenness with time instance graph
	void computeByShortestPathT()
	{
		 vector<Tran_Node>& tran_nodes = Tran_Graph.Tran_Data_Nodes;
		 int max_ts = transformer.data_graph.max_ts;
		 int tran_nodes_size = tran_nodes.size();
		 BC.clear();
		 BC.resize(data_nodes_size);
		 vector<vector<int>> t_distance(THREAD_NUM, vector<int>(data_nodes_size, -1));
		 vector<vector<int>> t_tran_distance(THREAD_NUM, vector<int>(tran_nodes_size, -1));
		 vector<vector<double>> t_tmp_sigma(THREAD_NUM, vector<double>(tran_nodes_size, 0));
		 vector<vector<int>> t_total_sigma(THREAD_NUM, vector<int>(tran_nodes_size, 0));
		 vector<vector<bool>> t_visited_tran_node(THREAD_NUM, vector<bool>(tran_nodes_size, 0));
		 vector<vector<vector<int>>> t_pre(THREAD_NUM, vector<vector<int>>(tran_nodes_size, vector<int>()));
		 omp_set_num_threads(THREAD_NUM);
		 #pragma omp parallel for
		 for (int i = 0; i < data_nodes_size; i++)
		 {
		 	int cur_thread_num = omp_get_thread_num();
		 	vector<int>& distance = t_distance[cur_thread_num];
		 	vector<int>& tran_distance = t_tran_distance[cur_thread_num];
		 	vector<double>& tmp_sigma = t_tmp_sigma[cur_thread_num];
		 	vector<vector<int>>& pre = t_pre[cur_thread_num];
		 	unordered_set<int> visited_tran_node ;
		 	transform(distance.cbegin(), distance.cend(), distance.begin(),[](const auto l) { return -1; });
		 	transform(tran_distance.cbegin(), tran_distance.cend(), tran_distance.begin(),[](const auto l) { return -1; });
		 	transform(tmp_sigma.cbegin(), tmp_sigma.cend(), tmp_sigma.begin(),[](const auto l) { return 0; });
		 	transform(pre.cbegin(), pre.cend(), pre.begin(),[](const auto l) { return vector<int>(); });
		 	//transform(visited_tran_node.cbegin(), visited_tran_node.cend(), visited_tran_node.begin(),[](const auto l) { return 0; });
		 	deque<int> q;
		 	stack<int> stk;
		 	map<int, int> total_sigma;
		 	unordered_map<int, int> tran_sigma;
		 	unordered_map<int, double> tran_delta;
		 	Tran_Node& tran_root_node = tran_nodes[tran_root_seqs[i]];
		 	q.push_back(tran_root_node.seq);
		 	distance[tran_root_node.identity] = 0;
		 	tran_distance[tran_root_node.seq] = 0;
		 	tran_sigma[tran_root_node.seq] = 1;
		 	tmp_sigma[tran_root_node.seq] = 1;
		 	while (!q.empty())
		 	{
		 		int u = q.front();
		 		stk.push(u);
		 		q.pop_front();
		 		for (int j = 0; j < tran_nodes[u].adj_edge.size(); j++)
		 		{
		 			int u_ = tran_nodes[u].adj_edge[j].destination_seq;
		 			int t_ = tran_nodes[u_].timestamp;
		 			if (tran_distance[u_] == -1 || tran_distance[u] + 1 == tran_distance[u_])
		 			{
		 				tran_distance[u_] = tran_distance[u] + 1;
		 				tran_sigma[u_] += tmp_sigma[u];
		 				tmp_sigma[u_] += tmp_sigma[u];
		 				if (distance[tran_nodes[u_].identity] == -1)
		 				{
		 					distance[tran_nodes[u_].identity] = tran_distance[u_];
		 				}
		 				if (!visited_tran_node.count(u_))
		 				{
		 					q.push_back(u_);
		 					visited_tran_node.insert(u_);
		 				}
		 				pre[u_].push_back(u);
		 			}
		 		}
		 	}
		 	for (auto& s : tran_sigma)
		 	{
		 		if (tran_distance[s.first] == distance[tran_nodes[s.first].identity])
		 		{

		 			total_sigma[tran_nodes[s.first].identity] += s.second;
		 		}
		 	}
		 	while (!stk.empty())
		 	{
		 		int u_ = stk.top();
		 		stk.pop();
		 		for (int j = 0; j < pre[u_].size(); j++)
		 		{
		 			if (pre[u_][j] == tran_root_node.seq) continue;
		 			pair<int, int> ut = { pre[u_][j] , tran_nodes[pre[u_][j]].timestamp };
		 			if (tran_nodes[ut.first].identity == tran_nodes[u_].identity)
		 			{
		 				tran_delta[ut.first] += tran_delta[u_];
		 			}
		 			else
		 			{
		 				if (tran_distance[u_] == distance[tran_nodes[u_].identity])
		 				{
		 					tran_delta[ut.first] += ((double)(tran_sigma[ut.first]) / total_sigma[tran_nodes[u_].identity]);
		 				}
		 				tran_delta[ut.first] += (double)tran_delta[u_] * tran_sigma[ut.first] / tran_sigma[u_];
		 			}
		 		}
		 	}
		 	#pragma omp critical
		 	{
		 		for (auto& d : tran_delta)
		 		{
		 			BC[tran_nodes[d.first].identity] += d.second;
		 		}
		 	}
		 }
	}
	//compute shortest path and temporal betweenness by compression
	void computeByShortestPath()
	{
		vector<Cmp_Node>& cmp_nodes = Cmp_Graph.cmp_data_nodes;
		int max_ts = transformer.data_graph.max_ts;
		int cmp_nodes_size = cmp_nodes.size();
		BC.clear();
		BC.resize(data_nodes_size);
		vector<vector<int>> t_distance(THREAD_NUM, vector<int>(data_nodes_size, -1));
		vector<vector<int>> t_cmp_distance(THREAD_NUM, vector<int>(cmp_nodes_size, -1));
		vector<vector<int>> t_tmp_sigma(THREAD_NUM, vector<int>(cmp_nodes_size, 0));
		vector<vector<int>> t_total_sigma(THREAD_NUM, vector<int>(cmp_nodes.size(), 0));
		vector<vector<vector<int>>> t_pre(THREAD_NUM, vector<vector<int>>(cmp_nodes_size, vector<int>()));
		vector<vector<int>> t_visited_cmp_node(THREAD_NUM, vector<int>(cmp_nodes_size, 0));
		omp_set_num_threads(THREAD_NUM);
		#pragma omp parallel for
		for (int i = 0; i < data_nodes_size; i++)
		{
			int cur_thread_num = omp_get_thread_num();
			vector<int>& distance = t_distance[cur_thread_num];
			vector<int>& cmp_distance = t_cmp_distance[cur_thread_num];
			vector<int>& tmp_sigma = t_tmp_sigma[cur_thread_num];
			vector<vector<int>>& pre = t_pre[cur_thread_num];
			vector<int>& visited_cmp_node = t_visited_cmp_node[cur_thread_num];
			Cmp_Node& cmp_root_node = cmp_nodes[i];
			transform(distance.cbegin(), distance.cend(), distance.begin(),[](const auto l) { return -1; });
			transform(cmp_distance.cbegin(), cmp_distance.cend(), cmp_distance.begin(),[](const auto l) { return -1; });
			transform(tmp_sigma.cbegin(), tmp_sigma.cend(), tmp_sigma.begin(),[](const auto l) { return 0; });
			transform(pre.cbegin(), pre.cend(), pre.begin(),[](const auto l) { return vector<int>(); });
			transform(visited_cmp_node.cbegin(), visited_cmp_node.cend(), visited_cmp_node.begin(),[](const auto l) { return 0; });
			unordered_map<int, int> total_sigma;
			unordered_map<int, int> cmp_sigma;
			unordered_map<int, double> cmp_delta;
			deque<int> q;
			stack<int> stk;
			q.push_back(cmp_root_node.seq);
			distance[cmp_root_node.identity] = 0;
			cmp_distance[cmp_root_node.seq] = 0;
			cmp_sigma[cmp_root_node.seq] = 1;
			tmp_sigma[cmp_root_node.seq] = 1;
			while (!q.empty())
			{
				int u = q.front();
				stk.push(u);
				q.pop_front();
				for (int j = 0; j < cmp_nodes[u].adj_edge.size(); j++)
				{
					int u_ = cmp_nodes[u].adj_edge[j].destination_seq;
					if (cmp_distance[u_] == -1 || cmp_distance[u] + 1 == cmp_distance[u_])
					{
						if (cmp_nodes[u].identity == cmp_nodes[u_].identity)
						{
							cmp_distance[u_] = cmp_distance[u];
							cmp_sigma[u_] += cmp_sigma[u];
							tmp_sigma[u_] += (cmp_sigma[u_] * cmp_nodes[u_].timestamps.size() + tmp_sigma[u]);
							if (!visited_cmp_node[u_])
							{
								q.push_front(u_);
								visited_cmp_node[u_] = 1;
							}
							for (int k = 0; k < pre[u].size(); k++)
							{
								if (cmp_nodes[pre[u][k]].identity != cmp_nodes[u].identity)
								{
									pre[u_].push_back(pre[u][k]);
								}
							}
						}
						else
						{
							cmp_distance[u_] = cmp_distance[u] + 1;
							cmp_sigma[u_] += tmp_sigma[u];
							tmp_sigma[u_] += tmp_sigma[u] * cmp_nodes[u_].timestamps.size();
							if (distance[cmp_nodes[u_].identity] == -1)
							{
								distance[cmp_nodes[u_].identity] = cmp_distance[u_];
							}
							if (!visited_cmp_node[u_])
							{
								q.push_back(u_);
								visited_cmp_node[u_] = 1;
							}
						}
						pre[u_].push_back(u);
					}
				}
			}
			for (auto& s : cmp_sigma)
			{
				if (cmp_distance[s.first] == distance[cmp_nodes[s.first].identity])
				{

					total_sigma[cmp_nodes[s.first].identity] += s.second * cmp_nodes[s.first].timestamps.size();
				}
			}
			while (!stk.empty())
			{
				int u_ = stk.top();
				stk.pop();
				for (int j = 0; j < pre[u_].size(); j++)
				{
					if (pre[u_][j] == cmp_root_node.seq) continue;
					pair<int, int> ut = { pre[u_][j] , cmp_nodes[pre[u_][j]].timestamps.front() };
					if (cmp_nodes[ut.first].identity == cmp_nodes[u_].identity)
					{
						cmp_delta[ut.first] += (cmp_delta[u_] * cmp_nodes[ut.first].timestamps.size() / cmp_nodes[u_].timestamps.size());
					}
					else
					{
						if (cmp_distance[u_] == distance[cmp_nodes[u_].identity])
						{
							cmp_delta[ut.first] += ((double)(cmp_sigma[ut.first] * cmp_nodes[u_].timestamps.size() * cmp_nodes[ut.first].timestamps.size()) / total_sigma[cmp_nodes[u_].identity]);
						}
						cmp_delta[ut.first] += (double)cmp_delta[u_] * cmp_sigma[ut.first] * cmp_nodes[ut.first].timestamps.size() / cmp_sigma[u_];
					}
				}
			}
			#pragma omp critical
			{
				for (auto& d : cmp_delta)
				{
					BC[cmp_nodes[d.first].identity] += d.second;
				}
			}
		}
	}
	//compute earliest path and temporal betweenness by compression
//	void computeByForemostPath()
//	{
//		vector<Cmp_Node>& cmp_nodes = Cmp_Graph.cmp_data_nodes;
//		int max_ts = transformer.data_graph.max_ts;
//		int cmp_nodes_size = cmp_nodes.size();
//		BC.clear();
//		BC.resize(data_nodes_size);
//		vector<vector<int>> t_fm_time(THREAD_NUM, vector<int>(data_nodes_size, 0));
//		vector<vector<int>> t_tmp_sigma(THREAD_NUM, vector<int>(cmp_nodes_size, 0));
//		vector<vector<int>> t_total_sigma(THREAD_NUM, vector<int>(cmp_nodes.size(), 0));
//		vector<vector<int>> t_bfs(THREAD_NUM, vector<int>(cmp_nodes.size(), -1));
//		vector<vector<vector<int>>> t_pre(THREAD_NUM, vector<vector<int>>(cmp_nodes_size, vector<int>()));
//		omp_set_num_threads(THREAD_NUM);
//		#pragma omp parallel for
//		for (int i = 0; i < data_nodes_size; i++)
//		{
//			int cur_thread_num = omp_get_thread_num();
//			Cmp_Node& cmp_root_node = cmp_nodes[i];
//			vector<int>& fm_time = t_fm_time[cur_thread_num];
//			transform(fm_time.cbegin(), fm_time.cend(), fm_time.begin(),[](const auto l) { return INT_MAX; });
//			unordered_set<int> visited_cmp;
//			queue<int> q;
//			for (int j = 0; j < cmp_root_node.adj_edge.size(); j++)
//			{
//				int des = cmp_root_node.adj_edge[j].destination_seq;
//				q.push(des);
//				visited_cmp.insert(des);
//				fm_time[cmp_nodes[des].identity] = min(fm_time[cmp_nodes[des].identity], cmp_nodes[des].timestamps[0]);
//			}
//			while (!q.empty())
//			{
//				int u = q.front();
//				q.pop();
//				for (int j = 0; j < cmp_nodes[u].adj_edge.size(); j++)
//				{
//					int des = cmp_nodes[u].adj_edge[j].destination_seq;
//					if (!visited_cmp.count(des) && cmp_nodes[u].identity != cmp_nodes[des].identity)
//					{
//						q.push(des);
//						visited_cmp.insert(des);
//						fm_time[cmp_nodes[des].identity] = min(fm_time[cmp_nodes[des].identity], cmp_nodes[des].timestamps[0]);
//					}
//				}
//			}
//			deque<pair<int, int>> dq;
//			stack<int> stk;
//			vector<int>& total_sigma = t_total_sigma[cur_thread_num];
//			unordered_map<int, int> cmp_sigma;
//			vector<int>& tmp_sigma = t_tmp_sigma[cur_thread_num];
//			unordered_map<int, int> update_tmp_sigma;
//			unordered_map<int, int> update_cmp_sigma;
//			vector<vector<int>>& pre = t_pre[cur_thread_num];
//			vector<int>& bfs = t_bfs[cur_thread_num];
//			unordered_map<int, double> cmp_delta;
//			transform(total_sigma.cbegin(), total_sigma.cend(), total_sigma.begin(),[](const auto l) { return 0; });
//			transform(tmp_sigma.cbegin(), tmp_sigma.cend(), tmp_sigma.begin(),[](const auto l) { return 0; });
//			transform(bfs.cbegin(), bfs.cend(), bfs.begin(),[](const auto l) { return -1; });
//			transform(pre.cbegin(), pre.cend(), pre.begin(),[](const auto l) { return vector<int>(); });
//			visited_cmp.clear();
//			dq.push_back({ cmp_root_node.seq ,0 });
//			cmp_sigma.insert({ cmp_root_node.seq,1 });
//			tmp_sigma[cmp_root_node.seq] = 1;
//			bfs[cmp_root_node.seq] = 0;
//			int last = -1;
//			while (!dq.empty())
//			{
//				pair<int, int> u = dq.front();
//				if (last == u.first)
//				{
//					dq.pop_front();
//					continue;
//				}
//				last = u.first;
//				stk.push(u.first);
//				dq.pop_front();
//				for (int j = 0; j < cmp_nodes[u.first].adj_edge.size(); j++)
//				{
//					int u_ = cmp_nodes[u.first].adj_edge[j].destination_seq;
//					if (u.second == 0)
//					{
//						if (bfs[u_] != -1 && bfs[u_] <= bfs[u.first])
//						{
//							cmp_sigma[u_] += tmp_sigma[u.first];
//							update_cmp_sigma[u_] += tmp_sigma[u.first];
//							tmp_sigma[u_] += tmp_sigma[u.first] * cmp_nodes[u_].timestamps.size();
//							update_tmp_sigma[u_] += tmp_sigma[u.first] * cmp_nodes[u_].timestamps.size();
//							dq.push_back({ u_ ,1 });
//							if (fm_time[cmp_nodes[u_].identity] == cmp_nodes[u_].timestamps[0])
//							{
//								total_sigma[cmp_nodes[u_].identity] += tmp_sigma[u.first];
//							}
//							pre[u_].push_back(u.first);
//						}
//						else
//						{
//							if (cmp_nodes[u.first].identity == cmp_nodes[u_].identity && (bfs[u_] == -1 || bfs[u.first] == bfs[u_]))
//							{
//								bfs[u_] = bfs[u.first];
//								cmp_sigma[u_] += cmp_sigma[u.first];
//								tmp_sigma[u_] += cmp_sigma[u_] * cmp_nodes[u_].timestamps.size() + tmp_sigma[u.first];
//								if (!visited_cmp.count(u_))
//								{
//									visited_cmp.insert(u_);
//									dq.push_front({ u_ ,0 });
//								}
//								for (int k = 0; k < pre[u.first].size(); k++)
//								{
//									pre[u_].push_back(pre[u.first][k]);
//								}
//							}
//							else if (cmp_nodes[u.first].identity != cmp_nodes[u_].identity && (bfs[u_] == -1 || bfs[u.first] + 1 == bfs[u_]))
//							{
//								bfs[u_] = bfs[u.first] + 1;
//								cmp_sigma[u_] += tmp_sigma[u.first];
//								tmp_sigma[u_] += tmp_sigma[u.first] * cmp_nodes[u_].timestamps.size();
//								if (!visited_cmp.count(u_))
//								{
//									visited_cmp.insert(u_);
//									dq.push_back({ u_ ,0 });
//								}
//								if (fm_time[cmp_nodes[u_].identity] == cmp_nodes[u_].timestamps[0])
//								{
//									total_sigma[cmp_nodes[u_].identity] += tmp_sigma[u.first];
//								}
//							}
//							pre[u_].push_back(u.first);
//						}
//					}
//					else
//					{
//						if (cmp_nodes[u.first].identity == cmp_nodes[u_].identity && (bfs[u_] == -1 || bfs[u.first] == bfs[u_]))
//						{
//							cmp_sigma[u_] += update_cmp_sigma[u.first];
//							update_cmp_sigma[u_] = update_cmp_sigma[u.first];
//							tmp_sigma[u_] += update_cmp_sigma[u_] * cmp_nodes[u_].timestamps.size() + update_tmp_sigma[u.first];
//							update_tmp_sigma[u_] = update_cmp_sigma[u_] * cmp_nodes[u_].timestamps.size() + update_tmp_sigma[u.first];
//							dq.push_front({ u_ ,1 });
//						}
//						else if (cmp_nodes[u.first].identity != cmp_nodes[u_].identity && (bfs[u_] == -1 || bfs[u.first] + 1 == bfs[u_]))
//						{
//							cmp_sigma[u_] += update_tmp_sigma[u.first];
//							update_cmp_sigma[u_] += update_tmp_sigma[u.first];
//							tmp_sigma[u_] += update_tmp_sigma[u.first] * cmp_nodes[u_].timestamps.size();
//							update_tmp_sigma[u_] += update_tmp_sigma[u.first] * cmp_nodes[u_].timestamps.size();
//							if (fm_time[cmp_nodes[u_].identity] == cmp_nodes[u_].timestamps[0])
//							{
//								total_sigma[cmp_nodes[u_].identity] += update_tmp_sigma[u.first];
//							}
//							dq.push_back({ u_ ,1 });
//						}
//					}
//					if (j == cmp_nodes[u.first].adj_edge.size() - 1)
//					{
//						update_cmp_sigma[u.first] = 0;
//						update_tmp_sigma[u.first] = 0;
//					}
//				}
//			}
//			visited_cmp.clear();
//			while (!stk.empty())
//			{
//				int u_ = stk.top();
//				stk.pop();
//				if (visited_cmp.count(u_)) continue;
//				visited_cmp.insert(u_);
//				bool flag = fm_time[cmp_nodes[u_].identity] == cmp_nodes[u_].timestamps[0];
//				for (int j = 0; j < pre[u_].size(); j++)
//				{
//					if (pre[u_][j] == cmp_root_node.seq) continue;
//					int u = pre[u_][j];
//					if (cmp_nodes[u].identity == cmp_nodes[u_].identity)
//					{
//						cmp_delta[u] += (cmp_delta[u_] * cmp_nodes[u].timestamps.size() / cmp_nodes[u_].timestamps.size());
//					}
//					else
//					{
//						if (flag)
//						{
//							cmp_delta[u] += ((double)(cmp_sigma[u] * cmp_nodes[u].timestamps.size()) / total_sigma[cmp_nodes[u_].identity]);
//						}
//						cmp_delta[u] += (double)cmp_delta[u_] * cmp_sigma[u] * cmp_nodes[u].timestamps.size() / cmp_sigma[u_];
//					}
//				}
//			}
//#pragma omp critical
//			{
//				for (auto& d : cmp_delta)
//				{
//					BC[cmp_nodes[d.first].identity] += d.second;
//				}
//			}
//		}
//	}
//	//compute earliest path and temporal betweenness with time instance graph
//	void computeByForemostPathT()
//	{
//		vector<Tran_Node>& tran_nodes = Tran_Graph.Tran_Data_Nodes;
//		int max_ts = transformer.data_graph.max_ts;
//		int tran_nodes_size = tran_nodes.size();
//		BC.clear();
//		BC.resize(data_nodes_size);
//		vector<vector<int>> t_fs_time(THREAD_NUM, vector<int>(data_nodes_size, 0));
//		vector<vector<int>> t_tmp_sigma(THREAD_NUM, vector<int>(tran_nodes_size, 0));
//		vector<vector<int>> t_total_sigma(THREAD_NUM, vector<int>(tran_nodes_size, 0));
//		vector<vector<int>> t_bfs(THREAD_NUM, vector<int>(tran_nodes_size, -1));
//		vector<vector<vector<int>>> t_pre(THREAD_NUM, vector<vector<int>>(tran_nodes_size, vector<int>()));
//		omp_set_num_threads(THREAD_NUM);
//		#pragma omp parallel for
//		for (int i = 0; i < data_nodes_size; i++)
//		{
//			int cur_thread_num = omp_get_thread_num();
//			Tran_Node& tran_root_node = tran_nodes[tran_root_seqs[i]];
//			vector<int>& fm_time = t_fs_time[cur_thread_num];
//			transform(fm_time.cbegin(), fm_time.cend(), fm_time.begin(),[](const auto l) { return INT_MAX; });
//			unordered_set<int> visited_tran;
//			queue<int> q;
//			for (int j = 0; j < tran_root_node.adj_edge.size(); j++)
//			{
//				int des = tran_root_node.adj_edge[j].destination_seq;
//				q.push(des);
//				visited_tran.insert(des);
//				fm_time[tran_nodes[des].identity] = min(fm_time[tran_nodes[des].identity], tran_nodes[des].timestamp);
//			}
//			while (!q.empty())
//			{
//				int u = q.front();
//				q.pop();
//				for (int j = 0; j < tran_nodes[u].adj_edge.size(); j++)
//				{
//					int des = tran_nodes[u].adj_edge[j].destination_seq;
//					if (!visited_tran.count(des) && tran_nodes[u].identity != tran_nodes[des].identity)
//					{
//						q.push(des);
//						visited_tran.insert(des);
//						fm_time[tran_nodes[des].identity] = min(fm_time[tran_nodes[des].identity], tran_nodes[des].timestamp);
//					}
//				}
//			}
//			deque<pair<int, int>> dq;//
//			stack<int> stk;
//			vector<int>& total_sigma = t_total_sigma[cur_thread_num];
//			unordered_map<int, int> tran_sigma;
//			vector<int>& tmp_sigma = t_tmp_sigma[cur_thread_num];
//			unordered_map<int, int> update_tmp_sigma;
//			unordered_map<int, int> update_tran_sigma;
//			vector<vector<int>>& pre = t_pre[cur_thread_num];
//			vector<int>& bfs = t_bfs[cur_thread_num];
//			unordered_map<int, double> tran_delta;
//			transform(total_sigma.cbegin(), total_sigma.cend(), total_sigma.begin(),[](const auto l) { return 0; });
//			transform(tmp_sigma.cbegin(), tmp_sigma.cend(), tmp_sigma.begin(),[](const auto l) { return 0; });
//			transform(bfs.cbegin(), bfs.cend(), bfs.begin(),[](const auto l) { return -1; });
//			transform(pre.cbegin(), pre.cend(), pre.begin(),[](const auto l) { return vector<int>(); });
//			visited_tran.clear();
//			dq.push_back({ tran_root_node.seq ,0 });
//			tran_sigma.insert({ tran_root_node.seq,1 });
//			tmp_sigma[tran_root_node.seq] = 1;
//			bfs[tran_root_node.seq] = 0;
//			int last = -1;
//			while (!dq.empty())
//			{
//				pair<int, int> u = dq.front();
//				if (last == u.first)//防止自环
//				{
//					dq.pop_front();
//					continue;
//				}
//				last = u.first;
//				stk.push(u.first);
//				dq.pop_front();
//				for (int j = 0; j < tran_nodes[u.first].adj_edge.size(); j++)
//				{
//					int u_ = tran_nodes[u.first].adj_edge[j].destination_seq;
//					if (u.second == 0)
//					{
//						if (bfs[u_] != -1 && bfs[u_] <= bfs[u.first])
//						{
//							tran_sigma[u_] += tmp_sigma[u.first];
//							update_tran_sigma[u_] += tmp_sigma[u.first];
//							tmp_sigma[u_] += tmp_sigma[u.first];
//							update_tmp_sigma[u_] += tmp_sigma[u.first];
//							dq.push_back({ u_ ,1 });
//							if (fm_time[tran_nodes[u_].identity] == tran_nodes[u_].timestamp)
//							{
//								total_sigma[tran_nodes[u_].identity] += tmp_sigma[u.first];
//							}
//							pre[u_].push_back(u.first);
//						}
//						else
//						{
//							bfs[u_] = bfs[u.first] + 1;
//							tran_sigma[u_] += tmp_sigma[u.first];
//							tmp_sigma[u_] += tmp_sigma[u.first];
//							if (!visited_tran.count(u_))
//							{
//								visited_tran.insert(u_);
//								dq.push_back({ u_ ,0});
//							}
//							if (fm_time[tran_nodes[u_].identity] == tran_nodes[u_].timestamp)
//							{
//								total_sigma[tran_nodes[u_].identity] += tmp_sigma[u.first];
//							}
//							pre[u_].push_back(u.first);
//						}
//					}
//					else
//					{
//						if (bfs[u_] == -1 || bfs[u.first] + 1 == bfs[u_])
//						{
//							tran_sigma[u_] += update_tmp_sigma[u.first];
//							update_tran_sigma[u_] += update_tmp_sigma[u.first];
//							tmp_sigma[u_] += update_tmp_sigma[u.first];
//							update_tmp_sigma[u_] += update_tmp_sigma[u.first];
//							if (fm_time[tran_nodes[u_].identity] == tran_nodes[u_].timestamp)
//							{
//								total_sigma[tran_nodes[u_].identity] += update_tmp_sigma[u.first];
//							}
//							dq.push_back({ u_ ,1 });
//						}
//
//					}
//				}
//				update_tran_sigma[u.first] = 0;
//				update_tmp_sigma[u.first] = 0;
//			}
//			visited_tran.clear();
//			while (!stk.empty())
//			{
//				int u_ = stk.top();
//				stk.pop();
//				if (visited_tran.count(u_)) continue;
//				visited_tran.insert(u_);
//				bool flag = fm_time[tran_nodes[u_].identity] == tran_nodes[u_].timestamp;
//				for (int j = 0; j < pre[u_].size(); j++)
//				{
//					if (pre[u_][j] == tran_root_node.seq) continue;
//					int u = pre[u_][j];
//					if (flag)
//					{
//						tran_delta[u] += ((double)(tran_sigma[u]) / total_sigma[tran_nodes[u_].identity]);
//					}
//					tran_delta[u] += (double)tran_delta[u_] * tran_sigma[u] / tran_sigma[u_];
//				}
//			}
//			#pragma omp critical
//			{
//				for (auto& d : tran_delta)
//				{
//					BC[tran_nodes[d.first].identity] += d.second;
//				}
//			}
//		}
//	}
	//compute shortest and earliest path and temporal betweenness by compression
	void computeByShortestForemostPath()
	{
		vector<Cmp_Node>& cmp_nodes = Cmp_Graph.cmp_data_nodes;
		int max_ts = transformer.data_graph.max_ts;
		int cmp_nodes_size = cmp_nodes.size();
		BC.resize(data_nodes_size);
		vector<vector<int>> t_distance(THREAD_NUM, vector<int>(data_nodes_size, -1));
		vector<vector<int>> t_cmp_distance(THREAD_NUM, vector<int>(cmp_nodes_size, -1));
		vector<vector<double>> t_tmp_sigma(THREAD_NUM, vector<double>(cmp_nodes_size, 0));
		vector<vector<double>> t_total_delta(THREAD_NUM, vector<double>(cmp_nodes.size(), 0));
		vector<vector<int>> t_total_sigma(THREAD_NUM, vector<int>(cmp_nodes.size(), 0));
		vector<vector<vector<int>>> t_pre(THREAD_NUM, vector<vector<int>>(cmp_nodes_size, vector<int>()));
		vector<vector<int>> t_fm_time(THREAD_NUM, vector<int>(data_nodes_size, INT_MAX));
		omp_set_num_threads(THREAD_NUM);
#pragma omp parallel for
		for (int i = 0; i < data_nodes_size; i++)
		{
			int cur_thread_num = omp_get_thread_num();
			vector<int>& distance = t_distance[cur_thread_num];
			vector<int>& cmp_distance = t_cmp_distance[cur_thread_num];
			vector<double>& tmp_sigma = t_tmp_sigma[cur_thread_num];
			vector<double>& total_delta = t_total_delta[cur_thread_num];
			vector<vector<int>>& pre = t_pre[cur_thread_num];
			vector<int>& fm_time = t_fm_time[cur_thread_num];
			Cmp_Node& cmp_root_node = cmp_nodes[i];
			transform(distance.cbegin(), distance.cend(), distance.begin(),[](const auto l) { return -1; });
			transform(cmp_distance.cbegin(), cmp_distance.cend(), cmp_distance.begin(),[](const auto l) { return -1; });
			transform(tmp_sigma.cbegin(), tmp_sigma.cend(), tmp_sigma.begin(),[](const auto l) { return 0; });
			transform(total_delta.cbegin(), total_delta.cend(), total_delta.begin(),[](const auto l) { return 0; });
			transform(pre.cbegin(), pre.cend(), pre.begin(),[](const auto l) { return vector<int>(); });
			transform(fm_time.cbegin(), fm_time.cend(), fm_time.begin(),[](const auto l) { return INT_MAX; });
			unordered_set<int> visited_cmp_node;
			unordered_map<int, int> cmp_sigma;
			unordered_map<int, double> cmp_delta;
			deque<int> q;
			stack<int> stk;
			q.push_back(cmp_root_node.seq);
			distance[cmp_root_node.identity] = 0;
			cmp_distance[cmp_root_node.seq] = 0;
			cmp_sigma[cmp_root_node.seq] = 1;
			tmp_sigma[cmp_root_node.seq] = 1;
			while (!q.empty())
			{
				int u = q.front();
				stk.push(u);
				fm_time[cmp_nodes[u].identity] = min(fm_time[cmp_nodes[u].identity], cmp_nodes[u].timestamps[0]);
				q.pop_front();
				for (int j = 0; j < cmp_nodes[u].adj_edge.size(); j++)
				{
					int u_ = cmp_nodes[u].adj_edge[j].destination_seq;
					if (cmp_distance[u_] == -1 || cmp_distance[u] + 1 == cmp_distance[u_])
					{
						if (cmp_nodes[u].identity == cmp_nodes[u_].identity)
						{
							cmp_distance[u_] = cmp_distance[u];
							cmp_sigma[u_] += cmp_sigma[u];
							tmp_sigma[u_] += (cmp_sigma[u_] * cmp_nodes[u_].timestamps.size() + tmp_sigma[u]);
							if (!visited_cmp_node.count(u_))
							{
								q.push_front(u_);
								visited_cmp_node.insert(u_);
							}
							for (int k = 0; k < pre[u].size(); k++)
							{
								if (cmp_nodes[pre[u][k]].identity != cmp_nodes[u].identity)
								{
									pre[u_].push_back(pre[u][k]);
								}
							}
						}
						else
						{
							cmp_distance[u_] = cmp_distance[u] + 1;
							cmp_sigma[u_] += tmp_sigma[u];
							tmp_sigma[u_] += tmp_sigma[u] * cmp_nodes[u_].timestamps.size();
							if (distance[cmp_nodes[u_].identity] == -1)
							{
								distance[cmp_nodes[u_].identity] = cmp_distance[u_];
							}
							if (!visited_cmp_node.count(u_))
							{
								q.push_back(u_);
								visited_cmp_node.insert(u_);
							}
						}
						pre[u_].push_back(u);
					}
				}
			}
			while (!stk.empty())
			{
				int u_ = stk.top();
				stk.pop();
				for (int j = 0; j < pre[u_].size(); j++)
				{
					if (pre[u_][j] == cmp_root_node.seq) continue;
					pair<int, int> ut = { pre[u_][j] , cmp_nodes[pre[u_][j]].timestamps.front() };
					int flag = fm_time[cmp_nodes[u_].identity] == cmp_nodes[u_].timestamps[0];
					if (cmp_nodes[ut.first].identity == cmp_nodes[u_].identity)
					{
						cmp_delta[ut.first] += (cmp_delta[u_] * cmp_nodes[ut.first].timestamps.size() / cmp_nodes[u_].timestamps.size());
					}
					else
					{
						if (flag)
						{
							cmp_delta[ut.first] += ((double)(cmp_sigma[ut.first] * cmp_nodes[ut.first].timestamps.size()) / cmp_sigma[u_]);
						}
						cmp_delta[ut.first] += (double)cmp_delta[u_] * cmp_sigma[ut.first] * cmp_nodes[ut.first].timestamps.size() / cmp_sigma[u_];
					}
				}
			}
			#pragma omp critical
			{
				for (auto& d : cmp_delta)
				{
					BC[cmp_nodes[d.first].identity] += d.second;
				}
			}
		}
	}
	//compute shortest and earliest path and temporal betweenness with time instance graph
	void computeByShortestForemostPathT()
	{
		vector<Tran_Node>& tran_nodes = Tran_Graph.Tran_Data_Nodes;
		int max_ts = transformer.data_graph.max_ts;
		int tran_nodes_size = tran_nodes.size();
		BC.resize(data_nodes_size);
		vector<vector<int>> t_distance(THREAD_NUM, vector<int>(data_nodes_size, -1));
		vector<vector<int>> t_tran_distance(THREAD_NUM, vector<int>(tran_nodes_size, -1));
		vector<vector<double>> t_tmp_sigma(THREAD_NUM, vector<double>(tran_nodes_size, 0));
		vector<vector<vector<int>>> t_pre(THREAD_NUM, vector<vector<int>>(tran_nodes_size, vector<int>()));
		vector<vector<int>> t_fm_time(THREAD_NUM, vector<int>(data_nodes_size, INT_MAX));
		omp_set_num_threads(THREAD_NUM);
		#pragma omp parallel for
		for (int i = 0; i < data_nodes_size; i++)
		{
			int cur_thread_num = omp_get_thread_num();
			vector<int>& distance = t_distance[cur_thread_num];
			vector<int>& tran_distance = t_tran_distance[cur_thread_num];
			vector<double>& tmp_sigma = t_tmp_sigma[cur_thread_num];
			vector<vector<int>>& pre = t_pre[cur_thread_num];
			vector<int>& fm_time = t_fm_time[cur_thread_num];
			transform(distance.cbegin(), distance.cend(), distance.begin(),[](const auto l) { return -1; });
			transform(tran_distance.cbegin(), tran_distance.cend(), tran_distance.begin(),[](const auto l) { return -1; });
			transform(tmp_sigma.cbegin(), tmp_sigma.cend(), tmp_sigma.begin(),[](const auto l) { return 0; });
			transform(pre.cbegin(), pre.cend(), pre.begin(),[](const auto l) { return vector<int>(); });
			transform(fm_time.cbegin(), fm_time.cend(), fm_time.begin(),[](const auto l) { return INT_MAX; });
			deque<int> q;
			stack<int> stk;
			unordered_map<int, int> total_sigma;
			unordered_set<int> visited_cmp_node;
			unordered_map<int, int> tran_sigma;
			unordered_map<int, double> tran_delta;
			Tran_Node& tran_root_node = tran_nodes[tran_root_seqs[i]];
			q.push_back(tran_root_node.seq);
			distance[tran_root_node.identity] = 0;
			tran_distance[tran_root_node.seq] = 0;
			tran_sigma[tran_root_node.seq] = 1;
			tmp_sigma[tran_root_node.seq] = 1;
			while (!q.empty())
			{
				int u = q.front();
				stk.push(u);
				fm_time[tran_nodes[u].identity] = min(fm_time[tran_nodes[u].identity], tran_nodes[u].timestamp);
				q.pop_front();
				for (int j = 0; j < tran_nodes[u].adj_edge.size(); j++)
				{
					int u_ = tran_nodes[u].adj_edge[j].destination_seq;
					int t_ = tran_nodes[u_].timestamp;
					if (tran_distance[u_] == -1 || tran_distance[u] + 1 == tran_distance[u_])
					{
						tran_distance[u_] = tran_distance[u] + 1;
						tran_sigma[u_] += tmp_sigma[u];
						tmp_sigma[u_] += tmp_sigma[u];
						if (distance[tran_nodes[u_].identity] == -1)
						{
							distance[tran_nodes[u_].identity] = tran_distance[u_];
						}

						if (!visited_cmp_node.count(u_))
						{
							q.push_back(u_);
							visited_cmp_node.insert(u_);
						}
						pre[u_].push_back(u);
					}
				}
			}
			while (!stk.empty())
			{
				int u_ = stk.top();
				stk.pop();
				for (int j = 0; j < pre[u_].size(); j++)
				{
					if (pre[u_][j] == tran_root_node.seq) continue;
					pair<int, int> ut = { pre[u_][j] , tran_nodes[pre[u_][j]].timestamp };
					int flag = fm_time[tran_nodes[u_].identity] == tran_nodes[u_].timestamp;
					if (flag)
					{
						tran_delta[ut.first] += ((double)(tran_sigma[ut.first]) / tran_sigma[u_]);
					}
					tran_delta[ut.first] += (double)tran_delta[u_] * tran_sigma[ut.first] / tran_sigma[u_];
				}
			}
			#pragma omp critical
			{
				for (auto& d : tran_delta)
				{
					BC[tran_nodes[d.first].identity] += d.second;
				}
			}
		}
	}
	//compute earliest path and temporal betweenness with time instance graph
//	void computeByFastestPathT()
//	{
//		vector<Tran_Node>& tran_nodes = Tran_Graph.Tran_Data_Nodes;
//		int max_ts = transformer.data_graph.max_ts;
//		int tran_nodes_size = tran_nodes.size();
//		BC.clear();
//		BC.resize(data_nodes_size);
//		vector<vector<int>> t_fs_time(THREAD_NUM, vector<int>(data_nodes_size, 0));
//		vector<vector<int>> t_tmp_sigma(THREAD_NUM, vector<int>(tran_nodes_size, 0));
//		vector<vector<int>> t_total_sigma(THREAD_NUM, vector<int>(tran_nodes_size, 0));
//		vector<vector<int>> t_bfs(THREAD_NUM, vector<int>(tran_nodes_size, -1));
//		vector<vector<vector<int>>> t_pre(THREAD_NUM, vector<vector<int>>(tran_nodes_size, vector<int>()));
//		omp_set_num_threads(THREAD_NUM);
//		#pragma omp parallel for
//		for (int i = 0; i < data_nodes_size; i++)
//		{
//			int cur_thread_num = omp_get_thread_num();
//			Tran_Node& tran_root_node = tran_nodes[tran_root_seqs[i]];
//			vector<int>& fs_time = t_fs_time[cur_thread_num];
//			transform(fs_time.cbegin(), fs_time.cend(), fs_time.begin(),[](const auto l) { return INT_MAX; });
//			unordered_set<int> visited_tran;
//			queue<pair<int,int>> q;
//			for (int j = 0; j < tran_root_node.adj_edge.size(); j++)
//			{
//				int des = tran_root_node.adj_edge[j].destination_seq;
//				int ts = tran_nodes[des].timestamp;
//				q.push({des,ts});
//				visited_tran.insert(des);
//				fs_time[tran_nodes[des].identity] = 0;
//			}
//			while (!q.empty())
//			{
//				pair<int,int> ut = q.front();
//				q.pop();
//				for (int j = 0; j < tran_nodes[ut.first].adj_edge.size(); j++)
//				{
//					int des = tran_nodes[ut.first].adj_edge[j].destination_seq;
//					if (!visited_tran.count(des) && tran_nodes[ut.first].identity != tran_nodes[des].identity)
//					{
//						q.push({ des,ut.second });
//						visited_tran.insert(des);
//					}
//					fs_time[tran_nodes[des].identity] = min(fs_time[tran_nodes[des].identity], tran_nodes[des].timestamp - ut.second);
//				}
//			}
//			deque<pair<pair<int, int>,int>> dq;//
//			stack<pair<int,int>> stk;
//			vector<int>& total_sigma = t_total_sigma[cur_thread_num];
//			unordered_map<int, int> tran_sigma;
//			vector<int>& tmp_sigma = t_tmp_sigma[cur_thread_num];
//			unordered_map<int, int> update_tmp_sigma;
//			unordered_map<int, int> update_tran_sigma;
//			vector<vector<int>>& pre = t_pre[cur_thread_num];
//			vector<int>& bfs = t_bfs[cur_thread_num];
//			unordered_map<int, double> tran_delta;
//			transform(total_sigma.cbegin(), total_sigma.cend(), total_sigma.begin(), [](const auto l) { return 0; });
//			transform(tmp_sigma.cbegin(), tmp_sigma.cend(), tmp_sigma.begin(),[](const auto l) { return 0; });
//			transform(bfs.cbegin(), bfs.cend(), bfs.begin(),[](const auto l) { return -1; });
//			transform(pre.cbegin(), pre.cend(), pre.begin(),[](const auto l) { return vector<int>(); });
//			visited_tran.clear();
//			dq.push_back({ { tran_root_node.seq ,0 } ,INT_MAX});
//			tran_sigma.insert({ tran_root_node.seq,1 });
//			tmp_sigma[tran_root_node.seq] = 1;
//			bfs[tran_root_node.seq] = 0;
//			int last = -1;
//			while (!dq.empty())
//			{
//				pair<pair<int, int>,int> u = dq.front();
//				if (last == u.first.first)//防止自环
//				{
//					dq.pop_front();
//					continue;
//				}
//				last = u.first.first;
//				stk.push({ u.first.first,u.second });
//				dq.pop_front();
//				for (int j = 0; j < tran_nodes[u.first.first].adj_edge.size(); j++)
//				{
//					int u_ = tran_nodes[u.first.first].adj_edge[j].destination_seq;
//					if (u.first.second == 0)
//					{
//						if (bfs[u_] != -1 && bfs[u_] <= bfs[u.first.first])
//						{
//							tran_sigma[u_] += tmp_sigma[u.first.first];
//							update_tran_sigma[u_] += tmp_sigma[u.first.first];
//							tmp_sigma[u_] += tmp_sigma[u.first.first];
//							update_tmp_sigma[u_] += tmp_sigma[u.first.first];
//							dq.push_back({ { u_ ,1 } ,u.second});
//							if (fs_time[tran_nodes[u_].identity] == tran_nodes[u_].timestamp - u.second);
//							{
//								total_sigma[tran_nodes[u_].identity] += tmp_sigma[u.first.first];
//							}
//							pre[u_].push_back(u.first.first);
//						}
//						else
//						{
//							bfs[u_] = bfs[u.first.first] + 1;
//							tran_sigma[u_] += tmp_sigma[u.first.first];
//							tmp_sigma[u_] += tmp_sigma[u.first.first];
//							if (!visited_tran.count(u_))
//							{
//								int ts = u.second;
//								if (ts == INT_MAX) ts = tran_nodes[u_].timestamp;
//								visited_tran.insert(u_);
//								dq.push_back({ {u_,0},ts});
//							}
//							if (u.second == INT_MAX || fs_time[tran_nodes[u_].identity] == tran_nodes[u_].timestamp - u.second)
//							{
//								total_sigma[tran_nodes[u_].identity] += tmp_sigma[u.first.first];
//							}
//							pre[u_].push_back(u.first.first);
//						}
//					}
//					else
//					{
//						if (bfs[u_] == -1 || bfs[u.first.first] + 1 == bfs[u_])
//						{
//							tran_sigma[u_] += update_tmp_sigma[u.first.first];
//							update_tran_sigma[u_] += update_tmp_sigma[u.first.first];
//							tmp_sigma[u_] += update_tmp_sigma[u.first.first];
//							update_tmp_sigma[u_] += update_tmp_sigma[u.first.first];
//							if (fs_time[tran_nodes[u_].identity] == tran_nodes[u_].timestamp)
//							{
//								total_sigma[tran_nodes[u_].identity] += update_tmp_sigma[u.first.first];
//							}
//							dq.push_back({ { u_ ,1 },u.second });
//						}
//
//					}
//				}
//				update_tran_sigma[u.first.first] = 0;
//				update_tmp_sigma[u.first.first] = 0;
//			}
//			visited_tran.clear();
//			while (!stk.empty())
//			{
//				pair<int,int> u_ = stk.top();
//				stk.pop();
//				if (visited_tran.count(u_.first)) continue;
//				visited_tran.insert(u_.first);
//				bool flag = fs_time[tran_nodes[u_.first].identity] == (tran_nodes[u_.first].timestamp - u_.second);
//				for (int j = 0; j < pre[u_.first].size(); j++)
//				{
//					if (pre[u_.first][j] == tran_root_node.seq) continue;
//					int u = pre[u_.first][j];
//					if (flag)
//					{
//						tran_delta[u] += ((double)(tran_sigma[u]) / total_sigma[tran_nodes[u_.first].identity]);
//					}
//					tran_delta[u] += (double)tran_delta[u_.first] * tran_sigma[u] / tran_sigma[u_.first];
//				}
//			}
//			#pragma omp critical
//			{
//				for (auto& d : tran_delta)
//				{
//					BC[tran_nodes[d.first].identity] += d.second;
//				}
//			}
//		}
//	}
//	void computeByFastestPath()
//	{
//		vector<Cmp_Node>& cmp_nodes = Cmp_Graph.cmp_data_nodes;
//		int max_ts = transformer.data_graph.max_ts;
//		int cmp_nodes_size = cmp_nodes.size();
//		BC.clear();
//		BC.resize(data_nodes_size);
//		vector<vector<int>> t_fs_time(THREAD_NUM, vector<int>(data_nodes_size, 0));
//		vector<vector<int>> t_tmp_sigma(THREAD_NUM, vector<int>(cmp_nodes_size, 0));
//		vector<vector<int>> t_total_sigma(THREAD_NUM, vector<int>(cmp_nodes.size(), 0));
//		vector<vector<int>> t_bfs(THREAD_NUM, vector<int>(cmp_nodes.size(), -1));
//		vector<vector<vector<int>>> t_pre(THREAD_NUM, vector<vector<int>>(cmp_nodes_size, vector<int>()));
//		omp_set_num_threads(THREAD_NUM);
//		#pragma omp parallel for
//		for (int i = 0; i < data_nodes_size; i++)
//		{
//			int cur_thread_num = omp_get_thread_num();
//			Cmp_Node& cmp_root_node = cmp_nodes[i];
//			vector<int>& fs_time = t_fs_time[cur_thread_num];
//			transform(fs_time.cbegin(), fs_time.cend(), fs_time.begin(),[](const auto l) { return INT_MAX; });
//			unordered_set<int> visited_cmp;
//			queue<pair<int,int>> q;
//			for (int j = 0; j < cmp_root_node.adj_edge.size(); j++)
//			{
//				int des = cmp_root_node.adj_edge[j].destination_seq;
//				int ts = cmp_nodes[des].timestamps[0];
//				q.push({des,ts});
//				visited_cmp.insert(des);
//				fs_time[cmp_nodes[des].identity] = 0;
//			}
//			while (!q.empty())
//			{
//				pair<int,int> ut = q.front();
//				q.pop();
//				for (int j = 0; j < cmp_nodes[ut.first].adj_edge.size(); j++)
//				{
//					int des = cmp_nodes[ut.first].adj_edge[j].destination_seq;
//					if (!visited_cmp.count(des) && cmp_nodes[ut.first].identity != cmp_nodes[des].identity)
//					{
//						q.push({des,ut.second});
//						visited_cmp.insert(des);
//					}
//					fs_time[cmp_nodes[des].identity] = min(fs_time[cmp_nodes[des].identity], cmp_nodes[des].timestamps[0] - ut.second);
//				}
//			}
//			deque<pair<pair<int,int>,int>> dq;
//			stack<pair<int,int>> stk;
//			vector<int>& total_sigma = t_total_sigma[cur_thread_num];
//			unordered_map<int, int> cmp_sigma;
//			vector<int>& tmp_sigma = t_tmp_sigma[cur_thread_num];
//			unordered_map<int, int> update_tmp_sigma;
//			unordered_map<int, int> update_cmp_sigma;
//			vector<vector<int>>& pre = t_pre[cur_thread_num];
//			vector<int>& bfs = t_bfs[cur_thread_num];
//			unordered_map<int, double> cmp_delta;
//			transform(total_sigma.cbegin(), total_sigma.cend(), total_sigma.begin(),[](const auto l) { return 0; });
//			transform(tmp_sigma.cbegin(), tmp_sigma.cend(), tmp_sigma.begin(),[](const auto l) { return 0; });
//			transform(bfs.cbegin(), bfs.cend(), bfs.begin(),[](const auto l) { return -1; });
//			transform(pre.cbegin(), pre.cend(), pre.begin(),[](const auto l) { return vector<int>(); });
//			visited_cmp.clear();
//			dq.push_back({{ cmp_root_node.seq ,0 },INT_MAX});
//			cmp_sigma.insert({ cmp_root_node.seq,1 });
//			tmp_sigma[cmp_root_node.seq] = 1;
//			bfs[cmp_root_node.seq] = 0;
//			int last = -1;
//			while (!dq.empty())
//			{
//				pair<pair<int, int>,int> u = dq.front();
//				if (last == u.first.first)
//				{
//					dq.pop_front();
//					continue;
//				}
//				last = u.first.first;
//				stk.push({ u.first.first, u.second });
//				dq.pop_front();
//				for (int j = 0; j < cmp_nodes[u.first.first].adj_edge.size(); j++)
//				{
//					int u_ = cmp_nodes[u.first.first].adj_edge[j].destination_seq;
//					if (u.second == 0)
//					{
//						if (bfs[u_] != -1 && bfs[u_] <= bfs[u.first.first])
//						{
//							cmp_sigma[u_] += tmp_sigma[u.first.first];
//							update_cmp_sigma[u_] += tmp_sigma[u.first.first];
//							tmp_sigma[u_] += tmp_sigma[u.first.first] * cmp_nodes[u_].timestamps.size();
//							update_tmp_sigma[u_] += tmp_sigma[u.first.first] * cmp_nodes[u_].timestamps.size();
//							dq.push_back({ { u_ ,1 } ,u.second});
//							if (fs_time[cmp_nodes[u_].identity] == cmp_nodes[u_].timestamps[0])
//							{
//								total_sigma[cmp_nodes[u_].identity] += tmp_sigma[u.first.first];
//							}
//							pre[u_].push_back(u.first.first);
//						}
//						else
//						{
//							if (cmp_nodes[u.first.first].identity == cmp_nodes[u_].identity && (bfs[u_] == -1 || bfs[u.first.first] == bfs[u_]))
//							{
//								bfs[u_] = bfs[u.first.first];
//								cmp_sigma[u_] += cmp_sigma[u.first.first];
//								tmp_sigma[u_] += cmp_sigma[u_] * cmp_nodes[u_].timestamps.size() + tmp_sigma[u.first.first];
//								if (!visited_cmp.count(u_))
//								{
//									visited_cmp.insert(u_);
//									dq.push_front({ { u_ ,0 } ,u.second});
//								}
//								for (int k = 0; k < pre[u.first.first].size(); k++)
//								{
//									pre[u_].push_back(pre[u.first.first][k]);
//								}
//							}
//							else if (cmp_nodes[u.first.first].identity != cmp_nodes[u_].identity && (bfs[u_] == -1 || bfs[u.first.first] + 1 == bfs[u_]))
//							{
//								bfs[u_] = bfs[u.first.first] + 1;
//								cmp_sigma[u_] += tmp_sigma[u.first.first];
//								tmp_sigma[u_] += tmp_sigma[u.first.first] * cmp_nodes[u_].timestamps.size();
//								if (!visited_cmp.count(u_))
//								{
//									int ts = u.second;
//									if (ts == INT_MAX) ts = cmp_nodes[u_].timestamps[0];
//									visited_cmp.insert(u_);
//									dq.push_back({{ u_ ,0 }, ts});
//								}
//								if (fs_time[cmp_nodes[u_].identity] == cmp_nodes[u_].timestamps[0])
//								{
//									total_sigma[cmp_nodes[u_].identity] += tmp_sigma[u.first.first];
//								}
//							}
//							pre[u_].push_back(u.first.first);
//						}
//					}
//					else
//					{
//						if (cmp_nodes[u.first.first].identity == cmp_nodes[u_].identity && (bfs[u_] == -1 || bfs[u.first.first] == bfs[u_]))
//						{
//							cmp_sigma[u_] += update_cmp_sigma[u.first.first];
//							update_cmp_sigma[u_] = update_cmp_sigma[u.first.first];
//							tmp_sigma[u_] += update_cmp_sigma[u_] * cmp_nodes[u_].timestamps.size() + update_tmp_sigma[u.first.first];
//							update_tmp_sigma[u_] = update_cmp_sigma[u_] * cmp_nodes[u_].timestamps.size() + update_tmp_sigma[u.first.first];
//							dq.push_front({ { u_ ,1 } ,u.second});
//						}
//						else if (cmp_nodes[u.first.first].identity != cmp_nodes[u_].identity && (bfs[u_] == -1 || bfs[u.first.first] + 1 == bfs[u_]))
//						{
//							cmp_sigma[u_] += update_tmp_sigma[u.first.first];
//							update_cmp_sigma[u_] += update_tmp_sigma[u.first.first];
//							tmp_sigma[u_] += update_tmp_sigma[u.first.first] * cmp_nodes[u_].timestamps.size();
//							update_tmp_sigma[u_] += update_tmp_sigma[u.first.first] * cmp_nodes[u_].timestamps.size();
//							if (fs_time[cmp_nodes[u_].identity] == cmp_nodes[u_].timestamps[0])
//							{
//								total_sigma[cmp_nodes[u_].identity] += update_tmp_sigma[u.first.first];
//							}
//							dq.push_back({ { u_ ,1 } ,u.second});
//						}
//					}
//					if (j == cmp_nodes[u.first.first].adj_edge.size() - 1)
//					{
//						update_cmp_sigma[u.first.first] = 0;
//						update_tmp_sigma[u.first.first] = 0;
//					}
//				}
//			}
//			visited_cmp.clear();
//			while (!stk.empty())
//			{
//				pair<int,int> u_ = stk.top();
//				stk.pop();
//				if (visited_cmp.count(u_.first)) continue;
//				visited_cmp.insert(u_.first);
//				bool flag = fs_time[cmp_nodes[u_.first].identity] == (cmp_nodes[u_.first].timestamps[0] - u_.second);
//				for (int j = 0; j < pre[u_.first].size(); j++)
//				{
//					if (pre[u_.first][j] == cmp_root_node.seq) continue;
//					int u = pre[u_.first][j];
//					if (cmp_nodes[u].identity == cmp_nodes[u_.first].identity)
//					{
//						cmp_delta[u] += (cmp_delta[u_.first] * cmp_nodes[u].timestamps.size() / cmp_nodes[u_.first].timestamps.size());
//					}
//					else
//					{
//						if (flag)
//						{
//							cmp_delta[u] += ((double)(cmp_sigma[u] * cmp_nodes[u].timestamps.size()) / total_sigma[cmp_nodes[u_.first].identity]);
//						}
//						cmp_delta[u] += (double)cmp_delta[u_.first] * cmp_sigma[u] * cmp_nodes[u].timestamps.size() / cmp_sigma[u_.first];
//					}
//				}
//			}
//#pragma omp critical
//			{
//				for (auto& d : cmp_delta)
//				{
//					BC[cmp_nodes[d.first].identity] += d.second;
//				}
//			}
//		}
//	}
	void computeByForemostPath()
	{
	vector<Cmp_Node>& cmp_nodes = Cmp_Graph.cmp_data_nodes;
	int max_ts = transformer.data_graph.max_ts;
	int cmp_nodes_size = cmp_nodes.size();
	BC.clear();
	BC.resize(data_nodes_size);
	vector<vector<int>> t_fm_time(THREAD_NUM, vector<int>(data_nodes_size, 0));
	vector<vector<int>> t_tmp_sigma(THREAD_NUM, vector<int>(cmp_nodes_size, 0));
	vector<vector<int>> t_total_sigma(THREAD_NUM, vector<int>(cmp_nodes.size(), 0));
	vector<vector<int>> t_bfs(THREAD_NUM, vector<int>(cmp_nodes.size(), -1));
	vector<vector<vector<int>>> t_pre(THREAD_NUM, vector<vector<int>>(cmp_nodes_size, vector<int>()));
	omp_set_num_threads(THREAD_NUM);
	#pragma omp parallel for
	for (int i = 0; i < data_nodes_size; i++)
	{
		int cur_thread_num = omp_get_thread_num();
		Cmp_Node& cmp_root_node = cmp_nodes[i];
		vector<int>& fm_time = t_fm_time[cur_thread_num];
		transform(fm_time.cbegin(), fm_time.cend(), fm_time.begin(), [](const auto l) { return -1; });
		unordered_set<int> visited_cmp;
		priority_queue<pair<int, int>, vector<pair<int, int>>, Compare> q;
		stack<int> stk;
		vector<int>& total_sigma = t_total_sigma[cur_thread_num];
		unordered_map<int, int> cmp_sigma;
		vector<int>& tmp_sigma = t_tmp_sigma[cur_thread_num];
		unordered_map<int, int> update_tmp_sigma;
		unordered_map<int, int> update_cmp_sigma;
		vector<vector<int>>& pre = t_pre[cur_thread_num];
		vector<int>& bfs = t_bfs[cur_thread_num];
		unordered_map<int, double> cmp_delta;
		transform(total_sigma.cbegin(), total_sigma.cend(), total_sigma.begin(), [](const auto l) { return 0; });
		transform(tmp_sigma.cbegin(), tmp_sigma.cend(), tmp_sigma.begin(), [](const auto l) { return 0; });
		transform(bfs.cbegin(), bfs.cend(), bfs.begin(), [](const auto l) { return -1; });
		transform(pre.cbegin(), pre.cend(), pre.begin(), [](const auto l) { return vector<int>(); });
		cmp_sigma[cmp_root_node.seq] = 1;
		for (int j = 0; j < cmp_root_node.adj_edge.size(); j++)
		{
			int des = cmp_root_node.adj_edge[j].destination_seq;
			int ts = cmp_nodes[des].timestamps[0];
			q.push({ des,ts });
			bfs[des] = 0;
			cmp_sigma[des] = 1 * cmp_nodes[des].timestamps.size();
			pre[des].push_back(cmp_root_node.seq);
		}

		while (!q.empty())
		{
			pair<int, int> ut = q.top();
			q.pop();
			if (fm_time[cmp_nodes[ut.first].identity] == -1 || fm_time[cmp_nodes[ut.first].identity] == ut.second) {
				fm_time[cmp_nodes[ut.first].identity] = ut.second ;
				total_sigma[cmp_nodes[ut.first].identity] += cmp_sigma[ut.first];
				stk.push( ut.first);
			}
			for (int i = 0; i < cmp_nodes[ut.first].adj_edge.size(); i++)
			{
				pair<int, int> u_t_ = { cmp_nodes[ut.first].adj_edge[i].destination_seq, cmp_nodes[cmp_nodes[ut.first].adj_edge[i].destination_seq].timestamps[0] };
				cmp_sigma[u_t_.first] += cmp_sigma[ut.first] * cmp_nodes[ut.first].timestamps.size();
				pre[u_t_.first].push_back(ut.first);
				if (visited_cmp.count(u_t_.first)) {
					bfs[u_t_.first] = max(bfs[ut.first] + 1, bfs[u_t_.first]);
				}
				else
				{
					visited_cmp.insert(u_t_.first);
					q.push(u_t_);
					bfs[u_t_.first] = bfs[ut.first] + 1;

				}
			}

		}
		while (!stk.empty())
		{
			int u_ = stk.top();
			stk.pop();
			bool flag = fm_time[cmp_nodes[u_].identity] == cmp_nodes[u_].timestamps[0];
			for (int j = 0; j < pre[u_].size(); j++)
			{
				if (pre[u_][j] == cmp_root_node.seq) continue;
				int u = pre[u_][j];
				if (cmp_nodes[u].identity == cmp_nodes[u_].identity)
				{
					cmp_delta[u] += (cmp_delta[u_] * cmp_nodes[u].timestamps.size() / cmp_nodes[u_].timestamps.size());
				}
				else
				{
					if (flag)
					{
						cmp_delta[u] += ((double)(cmp_sigma[u] * cmp_nodes[u].timestamps.size()) / total_sigma[cmp_nodes[u_].identity]);
					}
					cmp_delta[u] += (double)cmp_delta[u_] * cmp_sigma[u] * cmp_nodes[u].timestamps.size() / cmp_sigma[u_];
				}
			}
		}
#pragma omp critical
		{
			for (auto& d : cmp_delta)
			{
				BC[cmp_nodes[d.first].identity] += d.second;
			}
		}
	}
}
//void computeByForemostPath()
//{
//	vector<Cmp_Node>& cmp_nodes = Cmp_Graph.cmp_data_nodes;
//	int max_ts = transformer.data_graph.max_ts;
//	int cmp_nodes_size = cmp_nodes.size();
//	BC.clear();
//	BC.resize(data_nodes_size);
//	vector<vector<int>> t_fm_time(THREAD_NUM, vector<int>(data_nodes_size, 0));
//	vector<vector<int>> t_tmp_sigma(THREAD_NUM, vector<int>(cmp_nodes_size, 0));
//	vector<vector<int>> t_total_sigma(THREAD_NUM, vector<int>(cmp_nodes.size(), 0));
//	vector<vector<int>> t_bfs(THREAD_NUM, vector<int>(cmp_nodes.size(), -1));
//	vector<vector<vector<int>>> t_pre(THREAD_NUM, vector<vector<int>>(cmp_nodes_size, vector<int>()));
//	omp_set_num_threads(THREAD_NUM);
//	#pragma omp parallel for
//	for (int i = 0; i < data_nodes_size; i++)
//	{
//		int cur_thread_num = omp_get_thread_num();
//		Cmp_Node& cmp_root_node = cmp_nodes[i];
//		vector<int>& fm_time = t_fm_time[cur_thread_num];
//		transform(fm_time.cbegin(), fm_time.cend(), fm_time.begin(), [](const auto l) { return INT_MAX; });
//		unordered_set<int> visited_cmp;
//		queue<int> q;
//		for (int j = 0; j < cmp_root_node.adj_edge.size(); j++)
//		{
//			int des = cmp_root_node.adj_edge[j].destination_seq;
//			q.push(des);
//			visited_cmp.insert(des);
//			fm_time[cmp_nodes[des].identity] = min(fm_time[cmp_nodes[des].identity], cmp_nodes[des].timestamps[0]);
//		}
//		while (!q.empty())
//		{
//			int u = q.front();
//			q.pop();
//			for (int j = 0; j < cmp_nodes[u].adj_edge.size(); j++)
//			{
//				int des = cmp_nodes[u].adj_edge[j].destination_seq;
//				if (!visited_cmp.count(des) && cmp_nodes[u].identity != cmp_nodes[des].identity)
//				{
//					q.push(des);
//					visited_cmp.insert(des);
//					fm_time[cmp_nodes[des].identity] = min(fm_time[cmp_nodes[des].identity], cmp_nodes[des].timestamps[0]);
//				}
//			}
//		}
//		deque<pair<int, int>> dq;
//		stack<int> stk;
//		vector<int>& total_sigma = t_total_sigma[cur_thread_num];
//		unordered_map<int, int> cmp_sigma;
//		vector<int>& tmp_sigma = t_tmp_sigma[cur_thread_num];
//		unordered_map<int, int> update_tmp_sigma;
//		unordered_map<int, int> update_cmp_sigma;
//		vector<vector<int>>& pre = t_pre[cur_thread_num];
//		vector<int>& bfs = t_bfs[cur_thread_num];
//		unordered_map<int, double> cmp_delta;
//		transform(total_sigma.cbegin(), total_sigma.cend(), total_sigma.begin(), [](const auto l) { return 0; });
//		transform(tmp_sigma.cbegin(), tmp_sigma.cend(), tmp_sigma.begin(), [](const auto l) { return 0; });
//		transform(bfs.cbegin(), bfs.cend(), bfs.begin(), [](const auto l) { return -1; });
//		transform(pre.cbegin(), pre.cend(), pre.begin(), [](const auto l) { return vector<int>(); });
//		visited_cmp.clear();
//		dq.push_back({ cmp_root_node.seq ,0 });
//		cmp_sigma.insert({ cmp_root_node.seq,1 });
//		tmp_sigma[cmp_root_node.seq] = 1;
//		bfs[cmp_root_node.seq] = 0;
//		int last = -1;
//		while (!dq.empty())
//		{
//			pair<int, int> u = dq.front();
//			if (last == u.first)
//			{
//				dq.pop_front();
//				continue;
//			}
//			last = u.first;
//			stk.push(u.first);
//			dq.pop_front();
//			for (int j = 0; j < cmp_nodes[u.first].adj_edge.size(); j++)
//			{
//				int u_ = cmp_nodes[u.first].adj_edge[j].destination_seq;
//				if (u.second == 0)
//				{
//					if (bfs[u_] != -1 && bfs[u_] <= bfs[u.first])
//					{
//						cmp_sigma[u_] += tmp_sigma[u.first];
//						update_cmp_sigma[u_] += tmp_sigma[u.first];
//						tmp_sigma[u_] += tmp_sigma[u.first] * cmp_nodes[u_].timestamps.size();
//						update_tmp_sigma[u_] += tmp_sigma[u.first] * cmp_nodes[u_].timestamps.size();
//						dq.push_back({ u_ ,1 });
//						if (fm_time[cmp_nodes[u_].identity] == cmp_nodes[u_].timestamps[0])
//						{
//							total_sigma[cmp_nodes[u_].identity] += tmp_sigma[u.first];
//						}
//						pre[u_].push_back(u.first);
//					}
//					else
//					{
//						if (cmp_nodes[u.first].identity == cmp_nodes[u_].identity && (bfs[u_] == -1 || bfs[u.first] == bfs[u_]))
//						{
//							bfs[u_] = bfs[u.first];
//							cmp_sigma[u_] += cmp_sigma[u.first];
//							tmp_sigma[u_] += cmp_sigma[u_] * cmp_nodes[u_].timestamps.size() + tmp_sigma[u.first];
//							if (!visited_cmp.count(u_))
//							{
//								visited_cmp.insert(u_);
//								dq.push_front({ u_ ,0 });
//							}
//							for (int k = 0; k < pre[u.first].size(); k++)
//							{
//								pre[u_].push_back(pre[u.first][k]);
//							}
//						}
//						else if (cmp_nodes[u.first].identity != cmp_nodes[u_].identity && (bfs[u_] == -1 || bfs[u.first] + 1 == bfs[u_]))
//						{
//							bfs[u_] = bfs[u.first] + 1;
//							cmp_sigma[u_] += tmp_sigma[u.first];
//							tmp_sigma[u_] += tmp_sigma[u.first] * cmp_nodes[u_].timestamps.size();
//							if (!visited_cmp.count(u_))
//							{
//								visited_cmp.insert(u_);
//								dq.push_back({ u_ ,0 });
//							}
//							if (fm_time[cmp_nodes[u_].identity] == cmp_nodes[u_].timestamps[0])
//							{
//								total_sigma[cmp_nodes[u_].identity] += tmp_sigma[u.first];
//							}
//						}
//						pre[u_].push_back(u.first);
//					}
//				}
//				else
//				{
//					if (cmp_nodes[u.first].identity == cmp_nodes[u_].identity && (bfs[u_] == -1 || bfs[u.first] == bfs[u_]))
//					{
//						cmp_sigma[u_] += update_cmp_sigma[u.first];
//						update_cmp_sigma[u_] = update_cmp_sigma[u.first];
//						tmp_sigma[u_] += update_cmp_sigma[u_] * cmp_nodes[u_].timestamps.size() + update_tmp_sigma[u.first];
//						update_tmp_sigma[u_] = update_cmp_sigma[u_] * cmp_nodes[u_].timestamps.size() + update_tmp_sigma[u.first];
//						dq.push_front({ u_ ,1 });
//					}
//					else if (cmp_nodes[u.first].identity != cmp_nodes[u_].identity && (bfs[u_] == -1 || bfs[u.first] + 1 == bfs[u_]))
//					{
//						cmp_sigma[u_] += update_tmp_sigma[u.first];
//						update_cmp_sigma[u_] += update_tmp_sigma[u.first];
//						tmp_sigma[u_] += update_tmp_sigma[u.first] * cmp_nodes[u_].timestamps.size();
//						update_tmp_sigma[u_] += update_tmp_sigma[u.first] * cmp_nodes[u_].timestamps.size();
//						if (fm_time[cmp_nodes[u_].identity] == cmp_nodes[u_].timestamps[0])
//						{
//							total_sigma[cmp_nodes[u_].identity] += update_tmp_sigma[u.first];
//						}
//						dq.push_back({ u_ ,1 });
//					}
//				}
//				if (j == cmp_nodes[u.first].adj_edge.size() - 1)
//				{
//					update_cmp_sigma[u.first] = 0;
//					update_tmp_sigma[u.first] = 0;
//				}
//			}
//		}
//		visited_cmp.clear();
//		while (!stk.empty())
//		{
//			int u_ = stk.top();
//			stk.pop();
//			if (visited_cmp.count(u_)) continue;
//			visited_cmp.insert(u_);
//			bool flag = fm_time[cmp_nodes[u_].identity] == cmp_nodes[u_].timestamps[0];
//			for (int j = 0; j < pre[u_].size(); j++)
//			{
//				if (pre[u_][j] == cmp_root_node.seq) continue;
//				int u = pre[u_][j];
//				if (cmp_nodes[u].identity == cmp_nodes[u_].identity)
//				{
//					cmp_delta[u] += (cmp_delta[u_] * cmp_nodes[u].timestamps.size() / cmp_nodes[u_].timestamps.size());
//				}
//				else
//				{
//					if (flag)
//					{
//						cmp_delta[u] += ((double)(cmp_sigma[u] * cmp_nodes[u].timestamps.size()) / total_sigma[cmp_nodes[u_].identity]);
//					}
//					cmp_delta[u] += (double)cmp_delta[u_] * cmp_sigma[u] * cmp_nodes[u].timestamps.size() / cmp_sigma[u_];
//				}
//			}
//		}
//#pragma omp critical
//		{
//			for (auto& d : cmp_delta)
//			{
//				BC[cmp_nodes[d.first].identity] += d.second;
//			}
//		}
//	}
//}
//compute earliest path and temporal betweenness with time instance graph
struct Compare {
	bool operator()(const pair<int, int>& a, const pair<int, int>& b) {
		return a.second > b.second;
	}
};
void computeByForemostPathT()
{	
		vector<Tran_Node>& tran_nodes = Tran_Graph.Tran_Data_Nodes;
		int max_ts = transformer.data_graph.max_ts;
		int tran_nodes_size = tran_nodes.size();
		BC.clear();
		BC.resize(data_nodes_size);
		vector<vector<int>> t_fm_time(THREAD_NUM, vector<int>(data_nodes_size, 0));
		vector<vector<int>> t_tmp_sigma(THREAD_NUM, vector<int>(tran_nodes_size, 0));
		vector<vector<int>> t_total_sigma(THREAD_NUM, vector<int>(tran_nodes_size, 0));
		vector<vector<int>> t_bfs(THREAD_NUM, vector<int>(tran_nodes_size, -1));
		vector<vector<vector<int>>> t_pre(THREAD_NUM, vector<vector<int>>(tran_nodes_size, vector<int>()));
		omp_set_num_threads(THREAD_NUM);
		#pragma omp parallel for
		for (int i = 0; i < data_nodes_size; i++)
		{
			int cur_thread_num = omp_get_thread_num();
			Tran_Node& tran_root_node = tran_nodes[tran_root_seqs[i]];
			stack<int> stk;
			vector<int>& total_sigma = t_total_sigma[cur_thread_num];
			unordered_map<int, int> tran_sigma;
			vector<int>& tmp_sigma = t_tmp_sigma[cur_thread_num];
			vector<vector<int>>& pre = t_pre[cur_thread_num];
			vector<int>& bfs = t_bfs[cur_thread_num];
			unordered_map<int, double> tran_delta;
			transform(total_sigma.cbegin(), total_sigma.cend(), total_sigma.begin(), [](const auto l) { return 0; });
			transform(tmp_sigma.cbegin(), tmp_sigma.cend(), tmp_sigma.begin(), [](const auto l) { return 0; });
			transform(bfs.cbegin(), bfs.cend(), bfs.begin(), [](const auto l) { return -1; });
			transform(pre.cbegin(), pre.cend(), pre.begin(), [](const auto l) { return vector<int>(); });
			vector<int>& fm_time = t_fm_time[cur_thread_num];
			transform(fm_time.cbegin(), fm_time.cend(), fm_time.begin(),[](const auto l) { return -1; });
			unordered_set<int> visited_tran;
			priority_queue<pair<int, int>, vector<pair<int, int>>, Compare> q;
			tran_sigma[tran_root_node.seq] = 1;
			for (int j = 0; j < tran_root_node.adj_edge.size(); j++)
			{
				int des = tran_root_node.adj_edge[j].destination_seq;
				int ts = tran_root_node.adj_edge[j].destination_seq;
				q.push({des,ts});
				bfs[des] = 0;
				tran_sigma[des] = 1;
			}
			
			while (!q.empty())
			{
				pair<int, int> ut = q.top();
				q.pop();
				if (fm_time[tran_nodes[ut.first].identity] == -1 || fm_time[tran_nodes[ut.first].identity] == ut.second) {
					fm_time[tran_nodes[ut.first].identity] = ut.second;
					total_sigma[tran_nodes[ut.first].identity] += tran_sigma[ut.first];
					stk.push(ut.first);
				}
				for (int i = 0; i < tran_nodes[ut.first].adj_edge.size(); i++)
				{
					pair<int, int> u_t_ = { tran_nodes[ut.first].adj_edge[i].destination_seq, tran_nodes[tran_nodes[ut.first].adj_edge[i].destination_seq].timestamp };
					tran_sigma[u_t_.first] += tran_sigma[ut.first];
					pre[u_t_.first].push_back(ut.first);
					if (visited_tran.count(u_t_.first)) {
						bfs[u_t_.first] = max(bfs[ut.first] + 1, bfs[u_t_.first]);
					}
					else
					{
						visited_tran.insert(u_t_.first);
						q.push(u_t_);
						bfs[u_t_.first] = bfs[ut.first] + 1;

					}
				}

			}

			while (!stk.empty())
			{
				int u_ = stk.top();
				stk.pop();
				bool flag = fm_time[tran_nodes[u_].identity] == tran_nodes[u_].timestamp;
				for (int j = 0; j < pre[u_].size(); j++)
				{
					if (pre[u_][j] == tran_root_node.seq) continue;
					int u = pre[u_][j];
					if (flag)
					{
						tran_delta[u] += ((double)(tran_sigma[u]) / total_sigma[tran_nodes[u_].identity]);
					}
					tran_delta[u] += (double)tran_delta[u_] * tran_sigma[u] / tran_sigma[u_];
				}
			}
			#pragma omp critical
			{
				for (auto& d : tran_delta)
				{
					BC[tran_nodes[d.first].identity] += d.second;
				}
			}
		}
}
struct Compare2 {
	bool operator()(const pair<pair<int, int>,int>& a, const pair<pair<int, int>,int>& b) {
		return a.first.second - a.second > b.first.second - b.second;
	}
};
void computeByFastestPathT()
{
	vector<Tran_Node>& tran_nodes = Tran_Graph.Tran_Data_Nodes;
	int max_ts = transformer.data_graph.max_ts;
	int tran_nodes_size = tran_nodes.size();
	BC.clear();
	BC.resize(data_nodes_size);
	vector<vector<int>> t_fs_time(THREAD_NUM, vector<int>(data_nodes_size, 0));
	vector<vector<int>> t_tmp_sigma(THREAD_NUM, vector<int>(tran_nodes_size, 0));
	vector<vector<int>> t_total_sigma(THREAD_NUM, vector<int>(tran_nodes_size, 0));
	vector<vector<int>> t_bfs(THREAD_NUM, vector<int>(tran_nodes_size, -1));
	vector<vector<vector<int>>> t_pre(THREAD_NUM, vector<vector<int>>(tran_nodes_size, vector<int>()));
	omp_set_num_threads(THREAD_NUM);
	#pragma omp parallel for
	for (int i = 0; i < data_nodes_size; i++)
	{
		int cur_thread_num = omp_get_thread_num();
		Tran_Node& tran_root_node = tran_nodes[tran_root_seqs[i]];
		stack<pair<int,int>> stk;
		vector<int>& total_sigma = t_total_sigma[cur_thread_num];
		unordered_map<int, int> tran_sigma;
		vector<int>& tmp_sigma = t_tmp_sigma[cur_thread_num];
		vector<vector<int>>& pre = t_pre[cur_thread_num];
		vector<int>& bfs = t_bfs[cur_thread_num];
		unordered_map<int, double> tran_delta;
		transform(total_sigma.cbegin(), total_sigma.cend(), total_sigma.begin(), [](const auto l) { return 0; });
		transform(tmp_sigma.cbegin(), tmp_sigma.cend(), tmp_sigma.begin(), [](const auto l) { return 0; });
		transform(bfs.cbegin(), bfs.cend(), bfs.begin(), [](const auto l) { return -1; });
		transform(pre.cbegin(), pre.cend(), pre.begin(), [](const auto l) { return vector<int>(); });
		vector<int>& fs_time = t_fs_time[cur_thread_num];
		transform(fs_time.cbegin(), fs_time.cend(), fs_time.begin(), [](const auto l) { return -1; });
		unordered_set<int> visited_tran;
		priority_queue<pair<pair<int, int>,int>, vector<pair<pair<int, int>,int>>, Compare2> q;
		tran_sigma[tran_root_node.seq] = 1;
		for (int j = 0; j < tran_root_node.adj_edge.size(); j++)
		{
			int des = tran_root_node.adj_edge[j].destination_seq;
			int ts = tran_nodes[des].timestamp;
			q.push({ {des,ts}, ts});
			bfs[des] = 0;
			tran_sigma[des] = 1;
			pre[des].push_back(tran_root_node.seq);
		}

		while (!q.empty())
		{
			pair<pair<int, int>,int> ut = q.top();
			q.pop();
			if (fs_time[tran_nodes[ut.first.first].identity] == -1 || fs_time[tran_nodes[ut.first.first].identity] == ut.first.second - ut.second) {
				fs_time[tran_nodes[ut.first.first].identity] = ut.first.second - ut.second;
				total_sigma[tran_nodes[ut.first.first].identity] += tran_sigma[ut.first.first];
				stk.push({ ut.first.first, ut.second});
			}
			for (int i = 0; i < tran_nodes[ut.first.first].adj_edge.size(); i++)
			{
				pair<pair<int, int>, int> u_t_ = { { tran_nodes[ut.first.first].adj_edge[i].destination_seq, tran_nodes[tran_nodes[ut.first.first].adj_edge[i].destination_seq].timestamp } , ut.second};
				tran_sigma[u_t_.first.first] += tran_sigma[ut.first.first];
				pre[u_t_.first.first].push_back(ut.first.first);
				if (visited_tran.count(u_t_.first.first)) {
					bfs[u_t_.first.first] = max(bfs[ut.first.first] + 1, bfs[u_t_.first.first]);
				}
				else
				{
					visited_tran.insert(u_t_.first.first);
					q.push(u_t_);
					bfs[u_t_.first.first] = bfs[ut.first.first] + 1;

				}
			}

		}

		while (!stk.empty())
		{
			pair<int,int> u_ = stk.top();
			stk.pop();
			bool flag = fs_time[tran_nodes[u_.first].identity] == tran_nodes[u_.first].timestamp - u_.second;
			for (int j = 0; j < pre[u_.first].size(); j++)
			{
				if (pre[u_.first][j] == tran_root_node.seq) continue;
				int u = pre[u_.first][j];
				if (flag)
				{
					tran_delta[u] += ((double)(tran_sigma[u]) / total_sigma[tran_nodes[u_.first].identity]);
				}
				tran_delta[u] += (double)tran_delta[u_.first] * tran_sigma[u] / tran_sigma[u_.first];
			}
		}
		#pragma omp critical
		{
			for (auto& d : tran_delta)
			{
				BC[tran_nodes[d.first].identity] += d.second;
			}
		}
	}
}
void computeByFastestPath()
{
	vector<Cmp_Node>& cmp_nodes = Cmp_Graph.cmp_data_nodes;
	int max_ts = transformer.data_graph.max_ts;
	int cmp_nodes_size = cmp_nodes.size();
	BC.clear();
	BC.resize(data_nodes_size);
	vector<vector<int>> t_fs_time(THREAD_NUM, vector<int>(data_nodes_size, 0));
	vector<vector<int>> t_tmp_sigma(THREAD_NUM, vector<int>(cmp_nodes_size, 0));
	vector<vector<int>> t_total_sigma(THREAD_NUM, vector<int>(cmp_nodes.size(), 0));
	vector<vector<int>> t_bfs(THREAD_NUM, vector<int>(cmp_nodes.size(), -1));
	vector<vector<vector<int>>> t_pre(THREAD_NUM, vector<vector<int>>(cmp_nodes_size, vector<int>()));
	omp_set_num_threads(THREAD_NUM);
	#pragma omp parallel for
	for (int i = 0; i < data_nodes_size; i++)
	{
		int cur_thread_num = omp_get_thread_num();
		Cmp_Node& cmp_root_node = cmp_nodes[i];
		vector<int>& fs_time = t_fs_time[cur_thread_num];
		transform(fs_time.cbegin(), fs_time.cend(), fs_time.begin(),[](const auto l) { return -1; });
		unordered_set<int> visited_cmp;
		priority_queue<pair<pair<int, int>, int>, vector<pair<pair<int, int>, int>>, Compare2> q;
		stack<pair<int,int>> stk;
		vector<int>& total_sigma = t_total_sigma[cur_thread_num];
		unordered_map<int, int> cmp_sigma;
		vector<int>& tmp_sigma = t_tmp_sigma[cur_thread_num];
		unordered_map<int, int> update_tmp_sigma;
		unordered_map<int, int> update_cmp_sigma;
		vector<vector<int>>& pre = t_pre[cur_thread_num];
		vector<int>& bfs = t_bfs[cur_thread_num];
		unordered_map<int, double> cmp_delta;
		transform(total_sigma.cbegin(), total_sigma.cend(), total_sigma.begin(),[](const auto l) { return 0; });
		transform(tmp_sigma.cbegin(), tmp_sigma.cend(), tmp_sigma.begin(),[](const auto l) { return 0; });
		transform(bfs.cbegin(), bfs.cend(), bfs.begin(),[](const auto l) { return -1; });
		transform(pre.cbegin(), pre.cend(), pre.begin(),[](const auto l) { return vector<int>(); });
		cmp_sigma[cmp_root_node.seq] = 1;
		for (int j = 0; j < cmp_root_node.adj_edge.size(); j++)
		{
			int des = cmp_root_node.adj_edge[j].destination_seq;
			int ts = cmp_nodes[des].timestamps[0];
			q.push({ {des,ts}, ts });
			bfs[des] = 0;
			cmp_sigma[des] = 1 * cmp_nodes[des].timestamps.size();
			pre[des].push_back(cmp_root_node.seq);
		}

		while (!q.empty())
		{
			pair<pair<int, int>, int> ut = q.top();
			q.pop();
			if (fs_time[cmp_nodes[ut.first.first].identity] == -1 || fs_time[cmp_nodes[ut.first.first].identity] == ut.first.second - ut.second) {
				fs_time[cmp_nodes[ut.first.first].identity] = ut.first.second - ut.second;
				total_sigma[cmp_nodes[ut.first.first].identity] += cmp_sigma[ut.first.first];
				stk.push({ ut.first.first, ut.second });
			}
			for (int i = 0; i < cmp_nodes[ut.first.first].adj_edge.size(); i++)
			{
				pair<pair<int, int>, int> u_t_ = { { cmp_nodes[ut.first.first].adj_edge[i].destination_seq, cmp_nodes[cmp_nodes[ut.first.first].adj_edge[i].destination_seq].timestamps[0] } , ut.second };
				cmp_sigma[u_t_.first.first] += cmp_sigma[ut.first.first] * cmp_nodes[ut.first.first].timestamps.size();
				pre[u_t_.first.first].push_back(ut.first.first);
				if (visited_cmp.count(u_t_.first.first)) {
					bfs[u_t_.first.first] = max(bfs[ut.first.first] + 1, bfs[u_t_.first.first]);
				}
				else
				{
					visited_cmp.insert(u_t_.first.first);
					q.push(u_t_);
					bfs[u_t_.first.first] = bfs[ut.first.first] + 1;

				}
			}

		}
		while (!stk.empty())
		{
			pair<int,int> u_ = stk.top();
			stk.pop();
			bool flag = fs_time[cmp_nodes[u_.first].identity] == cmp_nodes[u_.first].timestamps[0] - u_.second;
			for (int j = 0; j < pre[u_.first].size(); j++)
			{
				if (pre[u_.first][j] == cmp_root_node.seq) continue;
				int u = pre[u_.first][j];
				if (cmp_nodes[u].identity == cmp_nodes[u_.first].identity)
				{
					cmp_delta[u] += (cmp_delta[u_.first] * cmp_nodes[u].timestamps.size() / cmp_nodes[u_.first].timestamps.size());
				}
				else
				{
					if (flag)
					{
						cmp_delta[u] += ((double)(cmp_sigma[u] * cmp_nodes[u].timestamps.size()) / total_sigma[cmp_nodes[u_.first].identity]);
					}
					cmp_delta[u] += (double)cmp_delta[u_.first] * cmp_sigma[u] * cmp_nodes[u].timestamps.size() / cmp_sigma[u_.first];
				}
			}
		}
		#pragma omp critical
		{
			for (auto& d : cmp_delta)
			{
				BC[cmp_nodes[d.first].identity] += d.second;
			}
		}
	}
}
};