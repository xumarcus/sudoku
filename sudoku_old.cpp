#include <iostream>
#include <algorithm>
#include <string>
#include <queue>
using namespace std;

int main() {
    queue <pair <int,int> >q;
    int val[9][9],ipos[9][9];
    int cell_count = 0;
    bool pos[9][9][10];
    string temp;
    cout << "Sudoku Solver running......" << endl;
    cout << "Please input your puzzle." << endl;
    for (int i=0; i<9; i++) {
        cout << "Line " << i+1 << ": ";
        cin >> temp;
        for (int j=0; j<9; j++) {
            val[i][j] = temp[j] - '0';
            if (val[i][j] != 0) q.push({i,j});
        }
    }
    cout << "Calculating..." << endl;
    for (int i=0; i<9; i++) {for (int j=0; j<9; j++) {for (int k=0; k<10; k++) pos[i][j][k] = true;}}
    for (int i=0; i<9; i++) {for (int j=0; j<9; j++) ipos[i][j] = 9;}
    while (!q.empty()) {
        int x = q.front().first;
        int y = q.front().second;
        int myval = val[x][y];
        int regx = 3*(x/3), regy = 3*(y/3);
        for (int i=0; i<9; i++) {
            if(pos[x][i][myval]) {
                pos[x][i][myval] = false;
                ipos[x][i]--;
            }
            if(ipos[x][i] == 1 && val[x][i] == 0) {
                for (int k=1; k<=9; k++) {
                    if (pos[x][i][k]) val[x][i] = k;
                }
            q.push({x,i});
            }
        }
        for (int i=0; i<9; i++) {
            if(pos[i][y][myval]) {
                pos[i][y][myval] = false;
                ipos[i][y]--;
            }
            if(ipos[i][y] == 1 && val[i][y] == 0) {
                for (int k=1; k<=9; k++) {
                    if (pos[i][y][k]) val[i][y] = k;
                }
            q.push({i,y});
            }
        }
        for (int i=regx; i<=regx+2; i++) {
            for (int j=regy; j<=regy+2; j++) {
                if(pos[i][j][myval]) {
                    pos[i][j][myval] = false;
                    ipos[i][j]--;
                }
                if(ipos[i][j] == 1 && val[i][j] == 0) {
                    for (int k=1; k<=9; k++) {
                        if (pos[i][j][k]) val[i][j] = k;
                    }
                    q.push({i,j});
                }
            }
        }
        q.pop();
        cell_count++;
    }
    cout << cell_count << " cells evaluated." << endl;
    cout << "Solution: " << endl;
    for (int i=0; i<9; i++) {for (int j=0; j<9; j++) {cout << val[i][j];} cout << endl;}
    cout << "Sudoku Solver ended." << endl;
}