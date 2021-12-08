#pragma once

#include "stdint.h"
using namespace std;

#ifdef B64
typedef uint64_t vid_t;
#elif B32
typedef uint32_t vid_t;
#endif

typedef float val_t;

struct neighbor_t {
    vid_t dst;
    val_t weight;
};

class csr_t {
 public:
    vid_t  v_count; //This is actual vcount in a graph 
    vid_t  e_count;
    vid_t  dst_size;
    vid_t* offset;
    neighbor_t* nebrs;
    int*  degrees;
    int64_t flag;

 public:
    csr_t() {
    };
    void init(vid_t a_vcount, vid_t a_dstsize, void* a_offset, void* a_nebrs, int64_t a_flag, vid_t edge_count) {
    }
    vid_t get_vcount() {
        return v_count;
    }
    vid_t get_ecount() {
        return e_count;
    }
    vid_t get_degree(vid_t index) {
        return offset[index + 1] - offset[index];
    }
};

class edge_t {
 public:
     vid_t src;
     vid_t dst;
     //edge properties here if any.
};

class coo_t {
 public:
     edge_t* edges;
     vid_t dst_size;
     vid_t v_count;
     vid_t e_count;
     coo_t() {
         edges = 0;
         dst_size = 0;
         v_count = 0;
         e_count = 0;
     }
     void init(vid_t a_vcount, vid_t a_dstsize, vid_t a_ecount, edge_t* a_edges) {
     }
};

class graph_t {
 public:
    csr_t csr;
    csr_t csc;
    coo_t coo;
 public:
    void init(vid_t a_vcount, vid_t a_dstsize, void* a_offset, void* a_nebrs, void* a_offset1, void* a_nebrs1, int64_t flag, int64_t num_vcount) {
        csr.offset = (vid_t*)a_offset;
        csr.nebrs = (neighbor_t*)a_nebrs;
        csc.offset = (vid_t*)a_offset1;
        csc.nebrs = (neighbor_t*)a_nebrs1;
        csr.v_count = a_vcount;
        csr.dst_size = a_dstsize;
        csr.flag = flag;
        csc.v_count = a_vcount;
        csc.dst_size = a_dstsize;
        csc.flag = flag;
    }

    vid_t get_vcount() {
        return csr.v_count;
    }
    vid_t get_edge_count() {
        return csr.e_count;
    }
};

