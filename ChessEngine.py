import numpy as np

class GameState():
    def __init__(self):
        self.board = [
            ['bR', 'bN', 'bB', 'bQ', 'bK', 'bB', 'bN', 'bR'],
            ['bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp'],
            ['--', '--', '--', '--', '--', '--', '--', '--'],
            ['--', '--', '--', '--', '--', '--', '--', '--'],
            ['--', '--', '--', '--', '--', '--', '--', '--'],
            ['--', '--', '--', '--', '--', '--', '--', '--'],
            ['wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp'],
            ['wR', 'wN', 'wB', 'wQ', 'wK', 'wB', 'wN', 'wR']
        ]
        self.moveFunction = {'p': self.getPawnMoves, 'R': self.getRookMoves, 'N': self.getKnightMoves,
                             'B': self.getBishopMoves, 'Q': self.getQueenMoves, 'K': self.getKingMoves}

        self.whiteToMove = True
        self.moveLog = []
        self.whiteKingLocation = (7, 4)
        self.blackKingLocation = (0, 4)
        self.checkMate = False
        self.staleMate = False
        self.insufficientMaterial = False # 기물 부족 무승부
        self.threefold_repetition = False # 동수 반복 무승부
        self.enpassantPossible = ()
        self.enpassantPossibleLog = [self.enpassantPossible]
        self.currentCastlingRight = CastleRights(True, True, True, True)
        self.castleRightsLog = [CastleRights(self.currentCastlingRight.wks, self.currentCastlingRight.bks,
                                             self.currentCastlingRight.wqs, self.currentCastlingRight.bqs)]
        self.repetition_history = {}
        self.repetition_history[self._get_position_hash()] = 1

    def _get_position_hash(self):
        # 현 상태서 각종 권한 정리
        board_tuple = tuple(map(tuple, self.board))
        castling_rights_tuple = (self.currentCastlingRight.wks, self.currentCastlingRight.wqs,
                                 self.currentCastlingRight.bks, self.currentCastlingRight.bqs)
        return (board_tuple, self.whiteToMove, castling_rights_tuple, self.enpassantPossible)

    def clone(self): # act용 딥커피 함수

        cloned_gs = GameState()
        cloned_gs.board = [row[:] for row in self.board]
        cloned_gs.whiteToMove = self.whiteToMove
        cloned_gs.moveLog = list(self.moveLog)
        cloned_gs.whiteKingLocation = self.whiteKingLocation
        cloned_gs.blackKingLocation = self.blackKingLocation
        cloned_gs.checkMate = self.checkMate
        cloned_gs.staleMate = self.staleMate
        cloned_gs.insufficientMaterial = self.insufficientMaterial
        cloned_gs.enpassantPossible = self.enpassantPossible
        cloned_gs.enpassantPossibleLog = list(self.enpassantPossibleLog)
        cloned_gs.currentCastlingRight = CastleRights(self.currentCastlingRight.wks, self.currentCastlingRight.bks,
                                                     self.currentCastlingRight.wqs, self.currentCastlingRight.bqs)
        cloned_gs.castleRightsLog = [CastleRights(cr.wks, cr.bks, cr.wqs, cr.bqs) for cr in self.castleRightsLog]
        cloned_gs.repetition_history = self.repetition_history.copy()
        return cloned_gs

    '''--- 기존 체스 게임 함수들 ---'''

    def makeMove(self, move, track_repetition=True):
        self.board[move.startRow][move.startCol] = '--'
        self.board[move.endRow][move.endCol] = move.pieceMoved
        self.moveLog.append(move)
        self.whiteToMove = not self.whiteToMove

        if move.pieceMoved == 'wK':
            self.whiteKingLocation = (move.endRow, move.endCol)
        elif move.pieceMoved == 'bK':
            self.blackKingLocation = (move.endRow, move.endCol)

        if move.isPawnPromotion:
            self.board[move.endRow][move.endCol] = move.pieceMoved[0] + 'Q'

        if move.isEnpassantMove:
            self.board[move.startRow][move.endCol] = '--'

        if move.pieceMoved[1] == 'p' and abs(move.startRow - move.endRow) == 2:
            self.enpassantPossible = ((move.startRow + move.endRow) // 2, move.endCol)
        else:
            self.enpassantPossible = ()

        if move.isCastleMove:
            if move.endCol - move.startCol == 2:
                self.board[move.endRow][move.endCol - 1] = self.board[move.endRow][move.endCol + 1]
                self.board[move.endRow][move.endCol + 1] = '--'
            else:
                self.board[move.endRow][move.endCol + 1] = self.board[move.endRow][move.endCol - 2]
                self.board[move.endRow][move.endCol - 2] = '--'

        self.enpassantPossibleLog.append(self.enpassantPossible)
        self.updateCastleRights(move)
        self.castleRightsLog.append(CastleRights(self.currentCastlingRight.wks, self.currentCastlingRight.bks,
                                                 self.currentCastlingRight.wqs, self.currentCastlingRight.bqs))
        
        if track_repetition:
            position_hash = self._get_position_hash()
            self.repetition_history[position_hash] = self.repetition_history.get(position_hash, 0) + 1
            if self.repetition_history[position_hash] >= 3:
                self.threefold_repetition = True
        
        return move.pieceCaptured

    def undoMove(self, track_repetition=True):
        if len(self.moveLog) != 0:
            if track_repetition:
                position_hash = self._get_position_hash()
                if position_hash in self.repetition_history:
                    self.repetition_history[position_hash] -= 1

            move = self.moveLog.pop()
            self.board[move.startRow][move.startCol] = move.pieceMoved
            self.board[move.endRow][move.endCol] = move.pieceCaptured
            self.whiteToMove = not self.whiteToMove

            if move.pieceMoved == 'wK':
                self.whiteKingLocation = (move.startRow, move.startCol)
            elif move.pieceMoved == 'bK':
                self.blackKingLocation = (move.startRow, move.startCol)

            if move.isEnpassantMove:
                self.board[move.endRow][move.endCol] = '--'
                self.board[move.startRow][move.endCol] = move.pieceCaptured

            self.enpassantPossibleLog.pop()
            self.enpassantPossible = self.enpassantPossibleLog[-1]

            self.castleRightsLog.pop()
            newRights = self.castleRightsLog[-1]
            self.currentCastlingRight = CastleRights(newRights.wks, newRights.bks, newRights.wqs, newRights.bqs)
            if move.isCastleMove:
                if move.endCol - move.startCol == 2:
                    self.board[move.endRow][move.endCol + 1] = self.board[move.endRow][move.endCol - 1]
                    self.board[move.endRow][move.endCol - 1] = '--'
                else:
                    self.board[move.endRow][move.endCol - 2] = self.board[move.endRow][move.endCol + 1]
                    self.board[move.endRow][move.endCol + 1] = '--'

            self.checkMate = False
            self.staleMate = False
            self.insufficientMaterial = False

    def updateCastleRights(self, move):
        if move.pieceMoved == 'wK':
            self.currentCastlingRight.wks = False
            self.currentCastlingRight.wqs = False
        elif move.pieceMoved == 'bK':
            self.currentCastlingRight.bks = False
            self.currentCastlingRight.bqs = False
        elif move.pieceMoved == 'wR':
            if move.startRow == 7:
                if move.startCol == 0:
                    self.currentCastlingRight.wqs = False
                elif move.startCol == 7:
                    self.currentCastlingRight.wks = False
        elif move.pieceMoved == 'bR':
            if move.startRow == 0:
                if move.startCol == 0:
                    self.currentCastlingRight.bqs = False
                elif move.startCol == 7:
                    self.currentCastlingRight.bks = False
        if move.pieceCaptured == 'wR':
            if move.endRow == 7:
                if move.endCol == 0:
                    self.currentCastlingRight.wqs = False
                elif move.endCol == 7:
                    self.currentCastlingRight.wks = False
        elif move.pieceCaptured == 'bR':
            if move.endRow == 0:
                if move.endCol == 0:
                    self.currentCastlingRight.bqs = False
                elif move.endCol == 7:
                    self.currentCastlingRight.bks = False

    def getValidMoves(self):
        tempEnpassantPossible = self.enpassantPossible
        tempCastlingRight = CastleRights(self.currentCastlingRight.wks, self.currentCastlingRight.bks,
                                         self.currentCastlingRight.wqs, self.currentCastlingRight.bqs)
        moves = self.getAllPossibleMoves()
        if self.whiteToMove:
            self.getCastleMoves(self.whiteKingLocation[0], self.whiteKingLocation[1], moves)
        else:
            self.getCastleMoves(self.blackKingLocation[0], self.blackKingLocation[1], moves)
        for i in range(len(moves) - 1, -1, -1):
            self.makeMove(moves[i], track_repetition=False)
            self.whiteToMove = not self.whiteToMove
            if self.inCheck():
                moves.remove(moves[i])
            self.whiteToMove = not self.whiteToMove
            self.undoMove(track_repetition=False)
        
        if len(moves) == 0:
            if self.inCheck():
                self.checkMate = True
            else:
                self.staleMate = True
        else:
            self.checkMate = False
            self.staleMate = False

        self.checkForInsufficientMaterial()

        self.enpassantPossible = tempEnpassantPossible
        self.currentCastlingRight = tempCastlingRight
        return moves

    def inCheck(self):
        if self.whiteToMove:
            return self.squardUnderAttack(self.whiteKingLocation[0], self.whiteKingLocation[1])
        else:
            return self.squardUnderAttack(self.blackKingLocation[0], self.blackKingLocation[1])

    def squardUnderAttack(self, r, c):
        self.whiteToMove = not self.whiteToMove
        oppMoves = self.getAllPossibleMoves()
        self.whiteToMove = not self.whiteToMove
        for move in oppMoves:
            if move.endRow == r and move.endCol == c:
                return True
        return False

    def getAllPossibleMoves(self):
        moves = []
        for r in range(len(self.board)):
            for c in range(len(self.board[r])):
                turn = self.board[r][c][0]
                if (turn == 'w' and self.whiteToMove) or (turn == 'b' and not self.whiteToMove):
                    piece = self.board[r][c][1]
                    self.moveFunction[piece](r, c, moves)
        return moves

    def getPawnMoves(self, r, c, moves):
        if self.whiteToMove:
            if self.board[r - 1][c] == '--':
                moves.append(Move((r, c), (r - 1, c), self.board))
                if r == 6 and self.board[r - 2][c] == '--':
                    moves.append(Move((r, c), (r - 2, c), self.board))

            if c - 1 >= 0:
                if self.board[r - 1][c - 1][0] == 'b':
                    moves.append(Move((r, c), (r - 1, c - 1), self.board))
                elif (r - 1, c - 1) == self.enpassantPossible:
                    moves.append(Move((r, c), (r - 1, c - 1), self.board, isEnpassantMove=True))

            if c + 1 <= 7:
                if self.board[r - 1][c + 1][0] == 'b':
                    moves.append(Move((r, c), (r - 1, c + 1), self.board))
                elif (r - 1, c + 1) == self.enpassantPossible:
                    moves.append(Move((r, c), (r - 1, c + 1), self.board, isEnpassantMove=True))
        else:
            if self.board[r + 1][c] == '--':
                moves.append(Move((r, c), (r + 1, c), self.board))
                if r == 1 and self.board[r + 2][c] == '--':
                    moves.append(Move((r, c), (r + 2, c), self.board))

            if c - 1 >= 0:
                if self.board[r + 1][c - 1][0] == 'w':
                    moves.append(Move((r, c), (r + 1, c - 1), self.board))
                elif (r + 1, c - 1) == self.enpassantPossible:
                    moves.append(Move((r, c), (r + 1, c - 1), self.board, isEnpassantMove=True))

            if c + 1 <= 7:
                if self.board[r + 1][c + 1][0] == 'w':
                    moves.append(Move((r, c), (r + 1, c + 1), self.board))
                elif (r + 1, c + 1) == self.enpassantPossible:
                    moves.append(Move((r, c), (r + 1, c + 1), self.board, isEnpassantMove=True))

    def getRookMoves(self, r, c, moves):
        directions = ((-1, 0), (0, -1), (1, 0), (0, 1))
        enemyColor = 'b' if self.whiteToMove else 'w'
        for d in directions:
            for i in range(1, 8):
                endRow = r + d[0] * i
                endCol = c + d[1] * i
                if 0 <= endRow < 8 and 0 <= endCol < 8:
                    endPiece = self.board[endRow][endCol]
                    if endPiece == '--':
                        moves.append(Move((r, c), (endRow, endCol), self.board))
                    elif endPiece[0] == enemyColor:
                        moves.append(Move((r, c), (endRow, endCol), self.board))
                        break
                    else:
                        break
                else:
                    break

    def getKnightMoves(self, r, c, moves):
        knightMoves = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))
        allyColor = 'w' if self.whiteToMove else 'b'
        for m in knightMoves:
            endRow = r + m[0]
            endCol = c + m[1]
            if 0 <= endRow < 8 and 0 <= endCol < 8:
                endPiece = self.board[endRow][endCol]
                if endPiece[0] != allyColor:
                    moves.append(Move((r, c), (endRow, endCol), self.board))

    def getBishopMoves(self, r, c, moves):
        directions = ((-1, -1), (-1, 1), (1, -1), (1, 1))
        enemyColor = 'b' if self.whiteToMove else 'w'
        for d in directions:
            for i in range(1, 8):
                endRow = r + d[0] * i
                endCol = c + d[1] * i
                if 0 <= endRow < 8 and 0 <= endCol < 8:
                    endPiece = self.board[endRow][endCol]
                    if endPiece == '--':
                        moves.append(Move((r, c), (endRow, endCol), self.board))
                    elif endPiece[0] == enemyColor:
                        moves.append(Move((r, c), (endRow, endCol), self.board))
                        break
                    else:
                        break
                else:
                    break

    def getQueenMoves(self, r, c, moves):
        self.getRookMoves(r, c, moves)
        self.getBishopMoves(r, c, moves)

    def getKingMoves(self, r, c, moves):
        kingMoves = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
        allyColor = 'w' if self.whiteToMove else 'b'
        for i in range(8):
            endRow = r + kingMoves[i][0]
            endCol = c + kingMoves[i][1]
            if 0 <= endRow < 8 and 0 <= endCol < 8:
                endPiece = self.board[endRow][endCol]
                if endPiece[0] != allyColor:
                    moves.append(Move((r, c), (endRow, endCol), self.board))

    def getCastleMoves(self, r, c, moves):
        if self.squardUnderAttack(r, c):
            return
        if (self.whiteToMove and self.currentCastlingRight.wks) or (
                not self.whiteToMove and self.currentCastlingRight.bks):
            self.getKingsideCastleMoves(r, c, moves)
        if (self.whiteToMove and self.currentCastlingRight.wqs) or (
                not self.whiteToMove and self.currentCastlingRight.bqs):
            self.getQueensideCastleMoves(r, c, moves)

    def getKingsideCastleMoves(self, r, c, moves):
        if self.board[r][c + 1] == '--' and self.board[r][c + 2] == '--':
            if not self.squardUnderAttack(r, c + 1) and not self.squardUnderAttack(r, c + 2):
                moves.append(Move((r, c), (r, c + 2), self.board, isCastleMove=True))

    def getQueensideCastleMoves(self, r, c, moves):
        if self.board[r][c - 1] == '--' and self.board[r][c - 2] == '--' and self.board[r][c - 3] == '--':
            if not self.squardUnderAttack(r, c - 1) and not self.squardUnderAttack(r, c - 2):
                moves.append(Move((r, c), (r, c - 2), self.board, isCastleMove=True))

    '''--- 기존 체스 게임 함수들 여기까지 ---'''

    def checkForInsufficientMaterial(self): # 기물 부족 무승부 판단
        for r in range(8):
            for c in range(8):
                piece_type = self.board[r][c][1]
                if piece_type in ('p', 'R', 'Q'):
                    self.insufficientMaterial = False
                    return

        white_knights = 0
        white_bishops = []
        black_knights = 0
        black_bishops = []

        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if piece != '--':
                    if piece == 'wN': white_knights += 1
                    elif piece == 'wB': white_bishops.append((r, c))
                    elif piece == 'bN': black_knights += 1
                    elif piece == 'bB': black_bishops.append((r, c))
        
        if white_knights == 0 and not white_bishops and black_knights == 0 and not black_bishops:
            self.insufficientMaterial = True
            return
            
        if (white_knights + len(white_bishops) == 0 and black_knights + len(black_bishops) == 1) or \
           (white_knights + len(white_bishops) == 1 and black_knights + len(black_bishops) == 0):
            self.insufficientMaterial = True
            return

        if len(white_bishops) == 1 and len(black_bishops) == 1 and white_knights == 0 and black_knights == 0:
            w_bishop_r, w_bishop_c = white_bishops[0]
            b_bishop_r, b_bishop_c = black_bishops[0]
            if (w_bishop_r + w_bishop_c) % 2 == (b_bishop_r + b_bishop_c) % 2:
                self.insufficientMaterial = True
                return
        
        self.insufficientMaterial = False

    def state_to_tensor(self): # 보드 현 상태 텐서로 변환
        tensor = np.zeros((18, 8, 8), dtype=np.float32)
        piece_to_channel = {
            'wp': 0, 'wN': 1, 'wB': 2, 'wR': 3, 'wQ': 4, 'wK': 5,
            'bp': 6, 'bN': 7, 'bB': 8, 'bR': 9, 'bQ': 10, 'bK': 11
        }
        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if piece != '--':
                    channel = piece_to_channel[piece]
                    tensor[channel, r, c] = 1
        if self.whiteToMove:
            tensor[12, :, :] = 1
        if self.currentCastlingRight.wks:
            tensor[13, :, :] = 1
        if self.currentCastlingRight.wqs:
            tensor[14, :, :] = 1
        if self.currentCastlingRight.bks:
            tensor[15, :, :] = 1
        if self.currentCastlingRight.bqs:
            tensor[16, :, :] = 1
        if self.enpassantPossible:
            r, c = self.enpassantPossible
            tensor[17, r, c] = 1
        return tensor

    '''
    총 텐서 18층, 8 * 8 사이즈 보드
    
    0층 : 백 폰
    1층 : 백 나이트
    2층 : 백 비숍
    3층 : 백 룩
    4층 : 백 퀸
    5층 : 백 킹
    6층 : 흑 폰
    7층 : 흑 나이트
    8층 : 흑 비숍
    9층 : 흑 룩
    10층 : 흑 퀸
    11층 : 흑 킹
    12층 : 어떤 색의 턴인지
    13층 : 백 킹사이드 캐슬링
    14층 : 백 퀸사이드 캐슬링
    15층 : 흑 킹사이드 캐슬링
    16층 : 흑 퀸사이드 캐슬링
    17층 : 앙파상 권한 
    '''

'''--- 기존 체스 게임 함수들 ---'''

class CastleRights():
    def __init__(self, wks, bks, wqs, bqs):
        self.wks = wks
        self.bks = bks
        self.wqs = wqs
        self.bqs = bqs


class Move():
    ranksToRows = {'1': 7, '2': 6, '3': 5, '4': 4,
                   '5': 3, '6': 2, '7': 1, '8': 0}
    rowsToRanks = {v: k for k, v in ranksToRows.items()}
    filesToCols = {'a': 0, 'b': 1, 'c': 2, 'd': 3,
                   'e': 4, 'f': 5, 'g': 6, 'h': 7}
    colsToFiles = {v: k for k, v in filesToCols.items()}

    def __init__(self, startSq, endSq, board, isEnpassantMove=False, isCastleMove=False):
        self.startRow = startSq[0]
        self.startCol = startSq[1]
        self.endRow = endSq[0]
        self.endCol = endSq[1]
        self.pieceMoved = board[self.startRow][self.startCol]
        self.pieceCaptured = board[self.endRow][self.endCol]
        self.isPawnPromotion = (self.pieceMoved == 'wp' and self.endRow == 0) or (
                self.pieceMoved == 'bp' and self.endRow == 7)
        self.isEnpassantMove = isEnpassantMove
        if self.isEnpassantMove:
            self.pieceCaptured = 'wp' if self.pieceMoved == 'bp' else 'bp'

        self.isCastleMove = isCastleMove
        self.isCapture = self.pieceCaptured != '--'
        
        # moveID를 0-4095 범위로 인코딩 (64*64-1)
        start_sq_idx = self.startRow * 8 + self.startCol
        end_sq_idx = self.endRow * 8 + self.endCol
        self.moveID = start_sq_idx * 64 + end_sq_idx

    def __eq__(self, other):
        if isinstance(other, Move):
            return self.moveID == other.moveID
        return False

    def getChessNotation(self):
        return self.getRankFile(self.startRow, self.startCol) + self.getRankFile(self.endRow, self.endCol)

    def getRankFile(self, r, c):
        return self.colsToFiles[c] + self.rowsToRanks[r]

    def __str__(self):
        if self.isCastleMove:
            return "O-O" if self.endCol == 6 else "O-O-O"
        endSquare = self.getRankFile(self.endRow, self.endCol)

        if self.pieceMoved[1] == 'p':
            if self.isCapture:
                return self.colsToFiles[self.startCol] + 'x' + endSquare
            else:
                return endSquare

        moveString = self.pieceMoved[-1]
        if self.isCapture:
            moveString += 'x'
        return moveString + endSquare

'''--- 기존 체스 게임 함수들 여기까지 ---'''