//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <gsl/gsl_rng.h>
#include "linelib.h"
#include "ransampl.h"

#define MAX_PATH_LENGTH 100

int binary = 0, num_threads = 1, vector_size = 100, negative = 5, second_order=1;
char job_id[MAX_STRING]="0";
long long samples = 1, edge_count_actual;
real alpha = 0.025, starting_alpha;

const gsl_rng_type * gsl_T;
gsl_rng * gsl_r;

line_node node_w, node_l, node_t, node_c;
line_hin hin_ww, hin_ll, hin_tt, hin_wl, hin_lt, hin_tw, hin_lw, hin_tl, hin_wt, hin_cl, hin_ct, hin_cw, hin_lc, hin_tc, hin_wc;
line_trainer_edge trainer_edge_ww, trainer_edge_ll, trainer_edge_tt, trainer_edge_wl, trainer_edge_lt, trainer_edge_tw, trainer_edge_lw, trainer_edge_tl, trainer_edge_wt, trainer_edge_cl, trainer_edge_ct, trainer_edge_cw, trainer_edge_lc, trainer_edge_tc, trainer_edge_wc;

double func_rand_num()
{
    return gsl_rng_uniform(gsl_r);
}

void *training_thread(void *id)
{
    long long edge_count = 0, last_edge_count = 0;
    unsigned long long next_random = (long long)id;
    real *error_vec = (real *)calloc(vector_size, sizeof(real));
    //int *node_lst = (int *)malloc(MAX_PATH_LENGTH * sizeof(int));
    
    while (1)
    {
        //judge for exit
        if (edge_count > samples / num_threads + 2) break;
        
        if (edge_count - last_edge_count > 1000)
        {
            edge_count_actual += edge_count - last_edge_count;
            last_edge_count = edge_count;
            // printf("%cAlpha: %f Progress: %.3lf%%", 13, alpha, (real)edge_count_actual / (real)(samples + 1) * 100);
            fflush(stdout);
            alpha = starting_alpha * (1 - edge_count_actual / (real)(samples + 1));
            if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
        }
        
        trainer_edge_ww.train_sample(alpha, error_vec, func_rand_num, next_random, second_order);
        trainer_edge_ll.train_sample(alpha, error_vec, func_rand_num, next_random, second_order);
        trainer_edge_tt.train_sample(alpha, error_vec, func_rand_num, next_random, second_order);
        trainer_edge_wl.train_sample(alpha, error_vec, func_rand_num, next_random, second_order);
        trainer_edge_lt.train_sample(alpha, error_vec, func_rand_num, next_random, second_order);
        trainer_edge_tw.train_sample(alpha, error_vec, func_rand_num, next_random, second_order);
        trainer_edge_lw.train_sample(alpha, error_vec, func_rand_num, next_random, second_order);
        trainer_edge_tl.train_sample(alpha, error_vec, func_rand_num, next_random, second_order);
        trainer_edge_wt.train_sample(alpha, error_vec, func_rand_num, next_random, second_order);
        trainer_edge_cl.train_sample(alpha, error_vec, func_rand_num, next_random, second_order);
        trainer_edge_ct.train_sample(alpha, error_vec, func_rand_num, next_random, second_order);
        trainer_edge_cw.train_sample(alpha, error_vec, func_rand_num, next_random, second_order);
        trainer_edge_lc.train_sample(alpha, error_vec, func_rand_num, next_random, second_order);
        trainer_edge_tc.train_sample(alpha, error_vec, func_rand_num, next_random, second_order);
        trainer_edge_wc.train_sample(alpha, error_vec, func_rand_num, next_random, second_order);
        
        edge_count += 1;
    }
    //free(node_lst);
    free(error_vec);
    pthread_exit(NULL);
}

void TrainModel() {
    strcat(job_id,".txt");
    char node_file_w[MAX_STRING]="node-w-", node_file_l[MAX_STRING]="node-l-", node_file_t[MAX_STRING]="node-t-", node_file_c[MAX_STRING]="node-c-";  
    char link_file_ww[MAX_STRING]="edge-ww-",link_file_ll[MAX_STRING]="edge-ll-",link_file_tt[MAX_STRING]="edge-tt-";
    char link_file_wl[MAX_STRING]="edge-wl-",link_file_lt[MAX_STRING]="edge-lt-",link_file_tw[MAX_STRING]="edge-tw-";
    char link_file_lw[MAX_STRING]="edge-lw-",link_file_tl[MAX_STRING]="edge-tl-",link_file_wt[MAX_STRING]="edge-wt-";
    char link_file_cl[MAX_STRING]="edge-cl-",link_file_ct[MAX_STRING]="edge-ct-",link_file_cw[MAX_STRING]="edge-cw-";
    char link_file_lc[MAX_STRING]="edge-lc-",link_file_tc[MAX_STRING]="edge-tc-",link_file_wc[MAX_STRING]="edge-wc-";
    char output_file_w[MAX_STRING]="output-w-", output_file_l[MAX_STRING]="output-l-", output_file_t[MAX_STRING]="output-t-", output_file_c[MAX_STRING]="output-c-";  
    char context_file_w[MAX_STRING]="context-w-", context_file_l[MAX_STRING]="context-l-", context_file_t[MAX_STRING]="context-t-", context_file_c[MAX_STRING]="context-c-";

    long a;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    starting_alpha = alpha;
    
    gsl_rng_env_setup();
    gsl_T = gsl_rng_rand48;
    gsl_r = gsl_rng_alloc(gsl_T);
    gsl_rng_set(gsl_r, 314159265);
    
    node_w.init(node_file_w, vector_size, job_id);
    node_l.init(node_file_l, vector_size, job_id);
    node_t.init(node_file_t, vector_size, job_id);
    node_c.init(node_file_c, vector_size, job_id);

    hin_ww.init(link_file_ww, &node_w, &node_w, job_id);
    hin_ll.init(link_file_ll, &node_l, &node_l, job_id);
    hin_tt.init(link_file_tt, &node_t, &node_t, job_id);
    hin_wl.init(link_file_wl, &node_w, &node_l, job_id);
    hin_lt.init(link_file_lt, &node_l, &node_t, job_id);
    hin_tw.init(link_file_tw, &node_t, &node_w, job_id);
    hin_lw.init(link_file_lw, &node_l, &node_w, job_id);
    hin_tl.init(link_file_tl, &node_t, &node_l, job_id);
    hin_wt.init(link_file_wt, &node_w, &node_t, job_id);    
    hin_cl.init(link_file_cl, &node_c, &node_l, job_id);
    hin_ct.init(link_file_ct, &node_c, &node_t, job_id);
    hin_cw.init(link_file_cw, &node_c, &node_w, job_id);
    hin_lc.init(link_file_lc, &node_l, &node_c, job_id);
    hin_tc.init(link_file_tc, &node_t, &node_c, job_id);
    hin_wc.init(link_file_wc, &node_w, &node_c, job_id);

    trainer_edge_ww.init('e', &hin_ww, negative);
    trainer_edge_ll.init('e', &hin_ll, negative);
    trainer_edge_tt.init('e', &hin_tt, negative);
    trainer_edge_wl.init('e', &hin_wl, negative);
    trainer_edge_lt.init('e', &hin_lt, negative);
    trainer_edge_tw.init('e', &hin_tw, negative);
    trainer_edge_lw.init('e', &hin_lw, negative);
    trainer_edge_tl.init('e', &hin_tl, negative);
    trainer_edge_wt.init('e', &hin_wt, negative);
    trainer_edge_cl.init('e', &hin_cl, negative);
    trainer_edge_ct.init('e', &hin_ct, negative);
    trainer_edge_cw.init('e', &hin_cw, negative);
    trainer_edge_lc.init('e', &hin_lc, negative);
    trainer_edge_tc.init('e', &hin_tc, negative);
    trainer_edge_wc.init('e', &hin_wc, negative);
    
//    clock_t start = clock();
    printf("Training...");
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, training_thread, (void *)a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    printf("\n");
//    clock_t finish = clock();
//    printf("Total time: %lf\n", (double)(finish - start) / CLOCKS_PER_SEC);
    
    node_w.output(output_file_w, binary, 0, job_id);
    node_l.output(output_file_l, binary, 0, job_id);
    node_t.output(output_file_t, binary, 0, job_id);
    node_c.output(output_file_c, binary, 0, job_id);
    node_w.output(context_file_w, binary, 1, job_id);
    node_l.output(context_file_l, binary, 1, job_id);
    node_t.output(context_file_t, binary, 1, job_id);
    node_c.output(context_file_c, binary, 1, job_id);
}

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

int main(int argc, char **argv) {
    int i;
    if (argc == 1) {
        printf("HIN2VEC\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-node <file>\n");
        printf("\t\tA dictionary of all nodes\n");
        printf("\t-link <file>\n");
        printf("\t\tAll links between nodes. Links are directed.\n");
        printf("\t-path <int>\n");
        printf("\t\tAll meta-paths. One path per line.\n");
        printf("\t-output <int>\n");
        printf("\t\tThe output file.\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
        printf("\t-size <int>\n");
        printf("\t\tSet size of word vectors; default is 100\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 5, common values are 5 - 10 (0 = not used)\n");
        printf("\t-samples <int>\n");
        printf("\t\tSet the number of training samples as <int>Million\n");
        printf("\t-iters <int>\n");
        printf("\t\tSet the number of interations.\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 1)\n");
        printf("\t-alpha <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025\n");
        printf("\nExamples:\n");
        printf("./hin2vec -node node.txt -link link.txt -path path.txt -output vec.emb -binary 1 -size 100 -negative 5 -samples 5 -iters 20 -threads 12\n\n");
        return 0;
    }
    // output_file[0] = 0;
    // if ((i = ArgPos((char *)"-node", argc, argv)) > 0) strcpy(node_file, argv[i + 1]);
    // if ((i = ArgPos((char *)"-link", argc, argv)) > 0) strcpy(link_file, argv[i + 1]);
    // if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) vector_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-samples", argc, argv)) > 0) samples = (long long)(atof(argv[i + 1])*1000000);
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-second_order", argc, argv)) > 0) second_order = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-job_id", argc, argv)) > 0) strcpy(job_id, argv[i + 1]);
    TrainModel();
    return 0;
}