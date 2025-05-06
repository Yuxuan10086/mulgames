import numpy as np
from collections import defaultdict
import random
from player import *
import csv
import os
import time  # 添加time模块导入
import math
# 使用tab作为缩进

def calculate_payoffs_continuous(x1: float, x2: float, V: int, d: float) -> tuple:
	"""
	计算连续博弈的收益（支持 x1, x2 ∈ [-1, 1]
	
	Args:
		x1, x2: 玩家的选择（[-1, 1] 区间实数）
		V: 货物价值（[1, 10000] 的整数）
		d: 耗损系数（[0.001, 0.5] 的浮点数）

	Returns:
		(payoff1, payoff2): 两玩家的收益
	"""
	# 处理 x1, x2 不在 [-1, 1] 区间的情况
	x1 = np.clip(x1, -1, 1)
	x2 = np.clip(x2, -1, 1)
	# 总收益函数
	g = V * ( (d + 1)/2 + (1 - d) * (x1 + x2) / 4 )
	
	# 处理分母为零的情况（双方完全合作时均分）
	denominator = 1 - x1 - x2
	if np.isclose(denominator, 0):
		return ((0.5 - x1) * g, (0.5 - x2) * g)
	
	# 按背叛程度分配收益
	payoff1 = (0.5 - x1) / denominator * g
	payoff2 = (0.5 - x2) / denominator * g
	return (payoff1, payoff2)

class Arena:
	"""擂台类：让所有玩家两两对局"""
	def __init__(self, players: list, rounds_per_match: int = 1000):
		self.players = players
		self.rounds_per_match = rounds_per_match
		# 存储每对玩家的对局结果
		self.match_results = {}

	def run(self):
		# 让每对玩家进行对局
		for i, player1 in enumerate(self.players):
			for j, player2 in enumerate(self.players[i+1:], i+1):
				# 创建一个只包含这两个玩家的模拟
				match = Simulation([player1, player2], self.rounds_per_match)
				match.run()
				
				# 记录对局结果
				player1_payoff = player1.get_total_payoff()
				player2_payoff = player2.get_total_payoff()
				self.match_results[(player1.player_id, player2.player_id)] = (player1_payoff, player2_payoff)
				
				# 重置玩家的记忆，为下一场对局做准备
				player1.memory.clear()
				player2.memory.clear()

	def get_ability_matrix(self) -> np.ndarray:
		"""计算并返回能力矩阵（收益比）"""
		# 获取所有玩家ID
		player_ids = sorted([p.player_id for p in self.players])
		n = len(player_ids)
		
		# 创建n*n的矩阵，对角线初始化为1
		ability_matrix = np.ones((n, n))
		
		# 填充矩阵
		for (p1_id, p2_id), (p1_payoff, p2_payoff) in self.match_results.items():
			# 获取玩家在矩阵中的索引
			i = player_ids.index(p1_id)
			j = player_ids.index(p2_id)
			
			# 处理分母为0的情况
			if p2_payoff == 0:
				ratio = float('inf') if p1_payoff > 0 else 1.0
			else:
				ratio = p1_payoff / p2_payoff
				
			# 填充矩阵对应位置
			ability_matrix[i][j] = round(ratio, 4)
			ability_matrix[j][i] = round(1 / ratio, 4) if ratio != 0 else float('inf')
		
		return ability_matrix

	def get_player_rankings(self) -> np.ndarray:
		"""计算并返回玩家能力排名"""
		# 获取能力矩阵
		ability_matrix = self.get_ability_matrix()
		
		# 计算每个玩家的平均能力值（排除对角线上的1）
		n = len(self.players)
		player_abilities = []
		
		for i, player in enumerate(self.players):
			# 获取该玩家的所有能力值（排除与自己的对局）
			abilities = ability_matrix[i]
			# 计算平均值时排除对角线上的1
			mean_ability = np.mean([v for j, v in enumerate(abilities) if i != j])
			player_abilities.append([player.player_id, mean_ability])
		
		# 转换为numpy数组并按能力值降序排序
		rankings = np.array(player_abilities)
		rankings = rankings[rankings[:, 1].argsort()[::-1]]
		
		return rankings

class Simulation:
	def __init__(self, players: list, max_rounds: int = 500000):
		self.players = players
		self.max_rounds = max_rounds
		self.bankruptcy_players = []  # 破产玩家列表

	def refresh(self):
		"""重置玩家状态"""
		for player in self.players:
			player.refresh()
		self.bankruptcy_players = []  # 重置破产玩家列表

	def run(self):
		for round_num in range(self.max_rounds):
			# 随机选择两名玩家
			a, b = random.sample(self.players, 2)
			if a.player_id in [bp["id"] for bp in self.bankruptcy_players] or \
				b.player_id in [bp["id"] for bp in self.bankruptcy_players]:
				continue  # 如果有玩家破产，跳过该轮
			
			# 生成对局参数
			V = int(np.clip(np.random.normal(5000, 2000), 1, 10000))
			d = np.clip(np.random.normal(0.25, 0.1), 0.001, 0.5)
			params = {"V": V, "d": d}
			
			# 玩家决策
			x_a = a.decide(b, params)
			x_b = b.decide(a, params)
			
			# 计算收益
			payoff_a, payoff_b = calculate_payoffs_continuous(x_a, x_b, V, d)
			
			# 记录对局结果
			a.record_interaction(b.player_id, params, x_a, x_b, payoff_a, payoff_b)
			b.record_interaction(a.player_id, params, x_b, x_a, payoff_b, payoff_a)

			# 检查破产
			if a.total_assets <= 0:
				self.bankruptcy_players.append({
					"id": a.player_id,
					"survival_rounds": round_num + 1
				})
			if b.total_assets <= 0:
				self.bankruptcy_players.append({
					"id": b.player_id,
					"survival_rounds": round_num + 1
				})
			
			if len(self.bankruptcy_players) == len(self.players) - 1:
				break  # 如果只剩一个玩家幸存，结束模拟

if __name__ == "__main__":
	# 创建玩家
	players = [TitForTatPlayer(1), AdaptiveBetrayer(2), RandomPlayer(3), AlwaysCooperatePlayer(4), \
		AlwaysDefectPlayer(5), GrimTriggerPlayer(6), WinStayLoseShiftPlayer(7), TitForTwoTatsPlayer(8), Prober(9)]

	sim = Simulation(players)
	round_num = 200
	survival_rounds = defaultdict(list)
	total_time = 0  # 记录总时间
	
	for i in range(round_num):
		print(f"第{i+1}轮模拟开始")
		start_time = time.time()  # 记录开始时间
		
		sim.run()
		
		end_time = time.time()  # 记录结束时间
		round_time = end_time - start_time  # 计算本轮用时
		total_time += round_time  # 累加总时间
		
		for player in players:
			print(f"玩家 {player.player_id} 的总资产: {player.total_assets:.2f}")
			bankruptcy_info = None
			for bp in sim.bankruptcy_players:
				if bp["id"] == player.player_id:
					bankruptcy_info = bp
					break
			if bankruptcy_info:
				print(f"存活时间: {bankruptcy_info['survival_rounds']} 回合后破产")
			else:
				print("存活时间: 存活到模拟结束")
			survival_rounds[player.player_id].append(bankruptcy_info['survival_rounds'] if bankruptcy_info else sim.max_rounds)
		sim.refresh()
		print(f"第{i+1}轮模拟结束，用时: {round_time:.2f}秒")
		print("-"*50)
	
	print(f"\n模拟总用时: {total_time:.2f}秒")
	print(f"平均每轮用时: {total_time/round_num:.2f}秒")
	print("-"*50)
	
	# 计算平均存活时间
	avg_survival_rounds = {player_id: sum(survival_rounds[player_id]) / round_num for player_id in survival_rounds}

	# 按平均存活时间排序
	sorted_players = sorted(avg_survival_rounds.items(), key=lambda x: x[1], reverse=True)

	# 输出排名
	print("玩家排名:")
	for rank, (player_id, avg_rounds) in enumerate(sorted_players, 1):
		print(f"排名 {rank}: 玩家 {player_id}, 平均存活回合数: {avg_rounds:.2f}")
	print("-"*50)
	# 将排名数据写入CSV文件
	
	# 准备写入的数据
	csv_data = []
	# 添加表头
	header = ['模拟轮数'] + [f'玩家{player_id}' for player_id in sorted(survival_rounds.keys())]
	csv_data.append(header)
	
	# 添加每轮的存活数据
	for round_idx in range(round_num):
		row = [round_idx + 1]  # 第一列为模拟轮数
		for player_id in sorted(survival_rounds.keys()):
			row.append(survival_rounds[player_id][round_idx])
		csv_data.append(row)
	
	# 确保输出目录存在
	output_dir = os.path.dirname(os.path.abspath(__file__))
	# 写入CSV文件
	output_file = os.path.join(output_dir, 'survival_rounds.csv')
	with open(output_file, 'w', newline='', encoding='utf-8') as f:
		writer = csv.writer(f)
		writer.writerows(csv_data)
	
	print(f"排名数据已保存至: {output_file}")
	
	print("\n"*3)

# todo: 加入“打听”机制 并行计算 加入拒绝博弈机制