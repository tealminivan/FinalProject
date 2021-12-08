#include <cassert>
#include <iostream>
#include <limits>
#include <list>
#include <vector>
#include <string.h>
#include <omp.h>

#include "kernel.h"

using std::cout;
using std::endl;

int THD_COUNT = 1;

using std::string;




void invoke_spmm(graph_t& graph, array2d_t<float> & input_array, array2d_t<float> & output_array)
{
    csr_t* snaph = &graph.csr;
    vid_t* offset_list = snaph-> offset;
    neighbor_t* nebrs_list = snaph-> nebrs;

    //create dense matrix from csr
    vector<vector<float>> matrix(input_array.row_count, vector<float>(input_array.row_count, 0));

    #pragma omp parallel for
    for (int v=0; v<snaph->v_count; v++){
        for (int neighbor = offset_list[v]; neighbor<offset_list[v+1]; neighbor++){
            matrix[v][nebrs_list[neighbor].dst]=nebrs_list[neighbor].weight;
        }
    }

    //matrix multiplication
    #pragma omp parallel for
    for (int i=0; i<input_array.row_count; ++i){
        for (int j=0; j<input_array.col_count; ++j){
            for (int k=0; k<input_array.row_count; ++k){
                output_array[i][j] += matrix[i][k] * input_array[k][j];
            }
        }
    }
    
    
}
