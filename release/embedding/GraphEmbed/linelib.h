#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <map>
#include <string>
#include <eigen3/Eigen/Dense>
#include "ransampl.h"
#include <iostream>

#define MAX_STRING 500
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
const int neg_table_size = 1e8;
const int hash_table_size = 30000000;

typedef float real;

typedef Eigen::Matrix< real, Eigen::Dynamic,
Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign >
BLPMatrix;

typedef Eigen::Matrix< real, 1, Eigen::Dynamic,
Eigen::RowMajor | Eigen::AutoAlign >
BLPVector;

struct struct_node {
    char *word;
};

struct hin_nb {
    int nb_id;
    double eg_wei;
    char eg_tp;
};

class line_node
{
protected:
    struct struct_node *node;
    int node_size, node_max_size, vector_size;
    char node_file[MAX_STRING];
    int *node_hash;
    real *_vec;
    real *_cvec;
    Eigen::Map<BLPMatrix> vec;
    Eigen::Map<BLPMatrix> cvec;
    
    int get_hash(char *word);
    int add_node(char *word);
public:
    line_node();
    ~line_node();
    
    friend class line_link;
    friend class line_hin;
    friend class line_trainer_edge;
    friend class line_trainer_path;
    
    void init(char *file_name, int vector_dim, char *job_id);
    int search(char *word);
    void output(char *file_name, int binary, int context, char *job_id);
    
    //friend void linelib_output_batch(char *file_name, int binary, line_node **array_line_node, int cnt);
};

class line_hin
{
protected:
    char hin_file[MAX_STRING];
    
    line_node *node_u, *node_v;
    std::vector<hin_nb> *hin;
    long long hin_size;
    
public:
    line_hin();
    ~line_hin();
    
    friend class line_trainer_edge;
    friend class line_trainer_path;
    
    void init(char *file_name, line_node *p_u, line_node *p_v, char *job_id);
};

class line_trainer_edge
{
protected:
    line_hin *phin;
    
    int *u_nb_cnt; int **u_nb_id; double **u_nb_wei;
    double *u_wei, *v_wei;
    ransampl_ws *smp_u, **smp_u_nb;
    real *expTable;
    int neg_samples, *neg_table;
    
    char edge_tp;
public:
    line_trainer_edge();
    ~line_trainer_edge();
    
    void init(char edge_type, line_hin *p_hin, int negative);
    void train_sample(real alpha, real *_error_vec, double (*func_rand_num)(), unsigned long long &rand_index, int second_order);
};
