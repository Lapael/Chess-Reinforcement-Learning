import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import math

class Node: # MCTS (Monte Carlo Tree Search)

    def __init__(self, parent=None, prior_p=1.0, move=None):
        self.parent = parent
        self.children = {}  # {move: Node}
        self.visit_count = 0
        self.total_action_value = 0  # Q-value
        self.prior_probability = prior_p  # P-value
        self.move = move # 이 노드로 오게 만든 move

    def expand(self, policy, valid_moves): # 노드에 대해 유효한 수를 추가함
        valid_move_set = {m.moveID for m in valid_moves}
        
        for move_id, prob in policy.items():
            # move_id를 실제 Move 객체로 변환해야 함
            # 이 부분은 select_action에서 처리
            if move_id in valid_move_set:
                self.children[move_id] = Node(parent=self, prior_p=prob, move=move_id)

    def select_child(self, c): # 점수 가장 높은거 선택
        return max(self.children.items(), key=lambda item: self._ucb_score(item[1], c))

    def update(self, value): # 역전파
        self.visit_count += 1
        self.total_action_value += value

    def _ucb_score(self, child, c): # 최대 점수 계산
        q_value = -child.total_action_value / (child.visit_count + 1e-8)
        
        # 부모의 방문 횟수를 이용해 탐험(exploration) 항 계산
        exploration_term = c * child.prior_probability * math.sqrt(self.visit_count) / (1 + child.visit_count)
        
        return q_value + exploration_term

    def is_leaf(self): # 리프노드 판단
        return not self.children




class PolicyValueNet(nn.Module): # 정책/가치 신경망 클래스

    def __init__(self):
        super(PolicyValueNet, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(18, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        # 정책 헤드
        self.policy_head_conv = nn.Conv2d(256, 2, kernel_size=1)
        self.policy_head_fc = nn.Linear(2 * 8 * 8, 4672)
        # 가치 헤드
        self.value_head_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.value_head_fc = nn.Sequential(
            nn.Linear(1 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_block(x)
        # 정책 헤드
        policy_out = self.policy_head_conv(x)
        policy_out = policy_out.view(-1, 2 * 8 * 8)
        policy_out = self.policy_head_fc(policy_out)
        # 가치 헤드
        value_out = self.value_head_conv(x)
        value_out = value_out.view(-1, 1 * 8 * 8)
        value_out = self.value_head_fc(value_out)
        return F.softmax(policy_out, dim=1), value_out



class Agent: # 에이전트 정의 클래스

    def __init__(self, model_path=None, c_puct=5, num_simulations=160, dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = PolicyValueNet().to(self.device) # 신경망 생성
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon # 변수들 ㅐㅈ정의

    def _get_policy_and_value(self, gs): # 게임 상태로 정책 가치 추출

        self.net.eval()
        state_tensor = torch.tensor(gs.state_to_tensor(), dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_probs, value = self.net(state_tensor)
        return policy_probs.cpu().numpy()[0], value.item()

    def select_action(self, gs, valid_moves, temperature=1.0): # MCTS로 수 선택
        # temperature은 수 선택 반영용 파라미터

        root = Node()

        policy_probs, _ = self._get_policy_and_value(gs)

        policy = {m.moveID: policy_probs[m.moveID] for m in valid_moves}

        # 디리클레 노이즈 추가
        if self.dirichlet_alpha > 0:
            num_valid_moves = len(valid_moves)
            dirichlet_noise = np.random.dirichlet([self.dirichlet_alpha] * num_valid_moves)
            
            # 노이즈를 정책에 적용
            noisy_policy = {}
            for i, (move_id, p) in enumerate(policy.items()):
                noisy_policy[move_id] = (1 - self.dirichlet_epsilon) * p + self.dirichlet_epsilon * dirichlet_noise[i]
            policy = noisy_policy

        prob_sum = sum(policy.values())
        if prob_sum > 0:
            policy = {move_id: p / prob_sum for move_id, p in policy.items()}
        else: # 모든 유효한 수의 확률이 0인 경우에는 균등 분배
            policy = {m.moveID: 1 / len(valid_moves) for m in valid_moves}

        root.expand(policy, valid_moves)

        for _ in range(self.num_simulations):
            self._mcts(gs.clone(), root) # 게임 상태 복사(딥카피)본으로 탐색

        # MCTS 결과로 얻은 정책(방문 횟수 분포) 생성
        mcts_policy = np.zeros(4672, dtype=np.float32)
        
        if not root.children:
            return None, mcts_policy

        # 온도 매개변수를 적용하여 수 선택
        child_visits = {move_id: child.visit_count for move_id, child in root.children.items()}
        
        if temperature == 0:
            # 가장 많이 방문한 수를 결정적으로 선택
            best_move_id = max(child_visits, key=child_visits.get)
        else:
            # 방문 횟수 분포에 따라 확률적으로 선택
            visits = np.array(list(child_visits.values()))
            move_ids = list(child_visits.keys())
            
            # 온도 적용
            visit_probs = np.power(visits, 1 / temperature)
            visit_probs /= np.sum(visit_probs)
            
            best_move_id = np.random.choice(move_ids, p=visit_probs)

        # MCTS 정책 계산 (학습 데이터용)
        total_visits = sum(child.visit_count for child in root.children.values())
        if total_visits > 0:
            for move_id, child_visit in child_visits.items():
                mcts_policy[move_id] = child_visit / total_visits

        # move_id를 Move 로
        best_move = None
        for move in valid_moves:
            if move.moveID == best_move_id:
                best_move = move
                break
        
        return best_move, mcts_policy

    def _create_move_from_id(self, board, move_id):
        #moveid로 move하기
        end_sq_idx = move_id % 64
        start_sq_idx = move_id // 64
        
        start_row = start_sq_idx // 8
        start_col = start_sq_idx % 8
        end_row = end_sq_idx // 8
        end_col = end_sq_idx % 8
        
        # ChessEngine의 Move 클래스를 직접 사용
        import ChessEngine
        return ChessEngine.Move((start_row, start_col), (end_row, end_col), board)

    def _mcts(self, gs, node):

        # 선택
        while not node.is_leaf():
            move_id, node = node.select_child(self.c_puct)
            # move_id로부터 Move 객체를 직접 생성하여 사용
            move = self._create_move_from_id(gs.board, move_id)
            gs.makeMove(move) # 가상으로 수 진행

        # 확장 평가
        policy_probs, value = self._get_policy_and_value(gs)
        
        valid_moves = gs.getValidMoves()
        if not valid_moves: # 게임 종료 상태
            value = -1.0 if gs.checkMate else 0.0 # 현재 플레이어 기준 패배는 -1, 무승부는 0
        else:
            # 유효한 수에 대해서만 정책을 필터링하고 정규화
            policy = {m.moveID: policy_probs[m.moveID] for m in valid_moves}
            prob_sum = sum(policy.values())
            if prob_sum > 0:
                policy = {move_id: p / prob_sum for move_id, p in policy.items()}
            else: # 모든 유효한 수의 확률이 0인 경우 (드묾), 균등 분배
                policy = {m.moveID: 1 / len(valid_moves) for m in valid_moves}
            node.expand(policy, valid_moves)

        # 역전파
        while node is not None:
            node.update(-value) # 상대방의 관점에서 가치를 업데이트
            value = -value
            node = node.parent

    def train_step(self, memory):
        # 정책 가치 망 학습 함수

        self.net.train()
        
        states = torch.tensor(np.array([data[0] for data in memory]), dtype=torch.float32).to(self.device)
        target_policies = torch.tensor(np.array([data[1] for data in memory]), dtype=torch.float32).to(self.device)
        target_values = torch.tensor(np.array([data[2] for data in memory]), dtype=torch.float32).to(self.device)

        predicted_policies, predicted_values = self.net(states)
        
        # Loss 계산
        policy_loss = -torch.sum(target_policies * torch.log(predicted_policies + 1e-8), dim=1).mean()
        value_loss = F.mse_loss(predicted_values.squeeze(), target_values)
        total_loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        print(f"Total Loss: {total_loss.item():.4f} (Policy: {policy_loss.item():.4f}, Value: {value_loss.item():.4f})")

    def save_model(self, path): # 모델 저장
        torch.save(self.net.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path): # 모델 로딩
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")