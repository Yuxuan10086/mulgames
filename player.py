import numpy as np
from collections import defaultdict
import random
import math
class Human:
	def __init__(self, player_id: int):
		self.player_id = player_id
		self.memory = defaultdict(list)  # 存储与每个对手的对局历史
		self.total_assets = 5000  # 初始总资产
		self.choice_history = []  # 存储选择历史
		self.expense = 20 * math.log(self.total_assets + 1, 2)

	def record_interaction(self, opponent_id: int, params: dict, 
						  my_choice: float, opponent_choice: float,
						  my_payoff: float, opponent_payoff: float):
		"""记录一次对局结果"""
		self.memory[opponent_id].append({
			"params": params,
			"my_choice": my_choice,
			"opponent_choice": opponent_choice,
			"my_payoff": my_payoff,
			"opponent_payoff": opponent_payoff
		})
		self.choice_history.append({
			"opponent_id": opponent_id,
			"params": params,
			"my_choice": my_choice,
			"opponent_choice": opponent_choice,
		})
		self.expense = 20 * math.log(self.total_assets + 1, 2)
		self.total_assets += my_payoff - self.expense

	def decide(self, opponent, params: dict) -> float:
		"""决策方法（需由子类实现）"""
		raise NotImplementedError

	def get_total_payoff(self) -> float:
		"""计算玩家的总收益"""
		total_payoff = sum([sum([h["my_payoff"] for h in history]) for history in self.memory.values()])
		return total_payoff
	
	def get_choice_history(self, opponent_id: int) -> list:
		"""获取与指定对手的所有选择记录"""
		return self.memory.get(opponent_id, [])
	
	def refresh(self):
		"""重置玩家状态"""
		self.memory = defaultdict(list)
		self.total_assets = 10000
		self.choice_history = []

class TitForTatPlayer(Human):
	"""以牙还牙策略（连续版）"""
	def decide(self, opponent, params: dict) -> float:
		history = self.memory.get(opponent.player_id, [])
		if not history:
			return 1.0  # 初始完全合作
		last_opponent_choice = history[-1]["opponent_choice"]
		return np.clip(last_opponent_choice, -1.0, 1.0)  # 模仿对手上次选择

class AdaptiveBetrayer(Human):
	"""自适应背叛策略：背叛程度与对手历史背叛均值负相关"""
	def decide(self, opponent, params: dict) -> float:
		history = self.memory.get(opponent.player_id, [])
		if not history:
			return 0.0  # 初始中等背叛
		avg_betrayal = np.mean([1 - h["opponent_choice"] for h in history])
		return np.clip(1 - avg_betrayal, -1.0, 1.0)

class RandomPlayer(Human):
	"""随机策略"""
	def decide(self, opponent, params: dict) -> float:
		return np.random.uniform(-1.0, 1.0)  # 随机选择一个 [-1, 1] 之间的数

class AlwaysCooperatePlayer(Human):
	"""永远合作策略"""
	def decide(self, opponent, params: dict) -> float:
		return 1.0

class AlwaysDefectPlayer(Human):
	"""永远背叛策略"""
	def decide(self, opponent, params: dict) -> float:
		return -1.0

class GrimTriggerPlayer(Human):
	"""冷酷触发策略"""
	def __init__(self, player_id: int):
		super().__init__(player_id)
		self.triggered = defaultdict(bool)  # 对手ID -> 是否触发永久背叛

	def decide(self, opponent, params: dict) -> float:
		if self.triggered[opponent.player_id]:
			return -1.0  # 永久背叛
		
		history = self.memory.get(opponent.player_id, [])
		if any(h["opponent_choice"] < 0 for h in history):
			self.triggered[opponent.player_id] = True
			return -1.0
		return 1.0  # 初始合作直到被背叛

class GenerousTitForTatPlayer(Human):
	"""宽容以牙还牙（30%概率原谅背叛）"""
	def decide(self, opponent, params: dict) -> float:
		history = self.memory.get(opponent.player_id, [])
		if not history:
			return 1.0
		
		last_opponent_choice = history[-1]["opponent_choice"]
		if last_opponent_choice < 0 and random.random() < 0.3:
			return 1.0  # 30%概率原谅
		return np.clip(last_opponent_choice, -1.0, 1.0)

class WinStayLoseShiftPlayer(Human):
	"""赢定输移策略（基于上一轮收益阈值）"""
	def __init__(self, player_id: int, threshold: float = 0.5):
		super().__init__(player_id)
		self.threshold = threshold  # 收益阈值（相对于V的比例）

	def decide(self, opponent, params: dict) -> float:
		history = self.memory.get(opponent.player_id, [])
		if not history:
			return 1.0
		
		last_entry = history[-1]
		expected = params["V"] * self.threshold
		if last_entry["my_payoff"] >= expected:
			return last_entry["my_choice"]  # 保持选择
		else:
			return -last_entry["my_choice"]  # 反向选择

class TitForTwoTatsPlayer(Human):
	"""两报还一报策略"""
	def decide(self, opponent, params: dict) -> float:
		history = self.memory.get(opponent.player_id, [])
		if len(history) < 2:
			return 1.0  # 前两轮默认合作
		
		# 检查最近两次是否都背叛
		last_two = [h["opponent_choice"] for h in history[-2:]]
		if all(c < 0 for c in last_two):
			return -1.0
		return 1.0

class Prober(Human):
	"""试探性策略（前3轮合作，第4轮背叛测试，根据反应调整）"""
	def __init__(self, player_id: int):
		super().__init__(player_id)
		self.probe_stage = defaultdict(int)  # 对手ID -> 试探阶段

	def decide(self, opponent, params: dict) -> float:
		stage = self.probe_stage[opponent.player_id]
		
		if stage < 3:
			self.probe_stage[opponent.player_id] += 1
			return 1.0  # 前3轮合作
		elif stage == 3:
			self.probe_stage[opponent.player_id] += 1
			return -1.0  # 第4轮试探性背叛
		else:
			# 分析对手对试探的反应
			history = self.memory.get(opponent.player_id, [])
			reaction = history[-1]["opponent_choice"] if history else 1.0
			return -1.0 if reaction < 0 else 1.0  # 对手报复则继续背叛