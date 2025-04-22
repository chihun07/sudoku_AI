import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

# 랜덤 시드 설정으로 재현성 확보
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 1. 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device.type.upper()} 사용 중.")

# 2. 모델 정의
class SudokuSolver(nn.Module):
    def __init__(self):
        super(SudokuSolver, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 9 * 9, 81 * 9)
        )

    def forward(self, x):
        x = self.model(x)
        return x.view(-1, 81, 9)

# 3. 데이터셋 클래스
class SudokuDataset(Dataset):
    def __init__(self, puzzles, solutions):
        self.puzzles = puzzles
        self.solutions = solutions

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        puzzle = torch.tensor(self.puzzles[idx], dtype=torch.float32).unsqueeze(0)  # [1,9,9]
        solution = torch.tensor(self.solutions[idx], dtype=torch.long)            # [81]
        return puzzle, solution

# 4. Sudoku 규칙 기반 손실 함수
def compute_rule_loss(probs):
    # probs: [batch, 81,9]
    batch_size = probs.size(0)
    probs = probs.view(batch_size, 9, 9, 9)  # [batch,row,col,num]
    loss = 0.0
    # 행 규칙: 각 숫자는 각 행에 한 번씩
    for i in range(9):
        row_sum = probs[:, i, :, :].sum(dim=1)  # [batch,num]
        loss += ((row_sum - 1)**2).mean()
    # 열 규칙
    for j in range(9):
        col_sum = probs[:, :, j, :].sum(dim=1)
        loss += ((col_sum - 1)**2).mean()
    # 블록 규칙
    for bi in range(3):
        for bj in range(3):
            block_probs = probs[:, bi*3:(bi+1)*3, bj*3:(bj+1)*3, :]
            blk_sum = block_probs.sum(dim=(1,2))  # [batch,num]
            loss += ((blk_sum - 1)**2).mean()
    return loss

# 5. 데이터 준비 및 모델 초기화

def generate_data():
    # 스도쿠 보드 초기화
    origin_board = np.zeros((9, 9), dtype=int)
    row = np.zeros((9, 10), dtype=int)
    col = np.zeros((9, 10), dtype=int)
    block = np.zeros((9, 10), dtype=int)

    # 백트래킹을 통한 완전한 스도쿠 생성
    def make_sudoku(k=0):
        if k > 80:
            return True
        i, j = divmod(k, 9)
        if origin_board[i, j] != 0:
            return make_sudoku(k + 1)

        nums = list(range(1, 10))
        random.shuffle(nums)
        b_idx = (i // 3) * 3 + (j // 3)
        for m in nums:
            if row[i, m] == 0 and col[j, m] == 0 and block[b_idx, m] == 0:
                row[i, m] = col[j, m] = block[b_idx, m] = 1
                origin_board[i, j] = m
                if make_sudoku(k + 1):
                    return True
                # 백트래킹
                row[i, m] = col[j, m] = block[b_idx, m] = 0
                origin_board[i, j] = 0
        return False

    make_sudoku()
    complete_board = origin_board.copy()

    # 숫자 제거하여 퍼즐 생성
    def remove_numbers(board, num_removed=40):
        b = board.copy()
        count = 0
        while count < num_removed:
            i, j = random.randint(0, 8), random.randint(0, 8)
            if b[i, j] != 0:
                b[i, j] = 0
                count += 1
        return b

    puzzle = remove_numbers(complete_board, num_removed=10)
    solution = complete_board
    # 솔루션을 크로마틱스에 맞춰 flatten
    solution_flat = [cell - 1 for r in solution for cell in r]
    return solution, puzzle, solution_flat

def prepare_dataset(sample_size=1000):
    puzzles, solutions = [], []
    for _ in range(sample_size):
        sol, puz, sol_flat = generate_data()
        puzzles.append(puz)
        solutions.append(sol_flat)
    return np.array(puzzles), np.array(solutions)

# 6. 모델·옵티마이저 구성 함수
def build_model(model_path="sudoku_model.pth", lr=5e-5, rule_weight=0.1):
    model = SudokuSolver().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    return model, criterion, optimizer, scheduler, model_path, rule_weight

# 7. 학습 함수 (규칙 손실 통합)
def train_model(model, train_loader, criterion, optimizer, scheduler, rule_weight, epochs=100, model_path="sudoku_model.pth"):
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        print("저장된 모델을 불러왔습니다.")
        return

    model.train()
    for epoch in range(1, epochs+1):
        total_loss = 0.0
        for puzzles_b, solutions_b in train_loader:
            puzzles_b, solutions_b = puzzles_b.to(device), solutions_b.to(device)
            optimizer.zero_grad()
            outputs = model(puzzles_b)  # [batch,81,9]
            ce_loss = criterion(outputs.view(-1,9), solutions_b.view(-1))
            probs = F.softmax(outputs, dim=2)
            rule_loss = compute_rule_loss(probs)
            loss = ce_loss + rule_weight * rule_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step(total_loss/len(train_loader))
        print(f"Epoch {epoch}/{epochs}, Total Loss: {total_loss/len(train_loader):.6f}")
    torch.save(model.state_dict(), model_path)
    model.eval()
    print("모델 학습 완료 및 저장되었습니다.")

# 8. 테스트 함수 (빈칸에만 적용)
def test_sudoku(model, num_tests=5):
    model.eval()
    for idx in range(num_tests):
        sol, puz, _ = generate_data()
        inp = torch.tensor(puz, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)
            pred = out.argmax(dim=2).view(9,9).cpu().numpy()+1
        solved = puz.copy()
        mask = (puz==0)
        solved[mask] = pred[mask]
        print(f"\n--- 테스트 {idx+1} ---")
        print("퍼즐:")
        print(puz)
        print("예측 적용 결과:")
        print(solved)
        print("정답:")
        print(sol)

# 9. 메인 실행
if __name__ == "__main__":
    puzzles, solutions = prepare_dataset(1000)
    dataset = SudokuDataset(puzzles, solutions)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=0)
    model, criterion, optimizer, scheduler, model_path, rule_weight = build_model()
    train_model(model, train_loader, criterion, optimizer, scheduler, rule_weight, epochs=100, model_path=model_path)
    test_sudoku(model, num_tests=5)
