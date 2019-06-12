/*
 * Copyright 2017 University of Sheffield.
 * Author: Peter Heywood 
 * Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
 *
 * University of Sheffield retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * University of Sheffield is strictly prohibited.
 *
 * For terms of licence agreement please attached licence or view licence 
 * on www.flamegpu.com website.
 * 
 */

#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

#include "header.h"
// #include <vector>


__FLAME_GPU_HOST_FUNC__ void generateAAgents(){
	unsigned int a_count = *get_A_POP();
	if( a_count > xmachine_memory_A_MAX){
		fprintf(stderr, "Error: %u A agents requested. Only room for %u\n", a_count, xmachine_memory_A_MAX);
		fflush(stderr);
		exit(EXIT_FAILURE);
	}

    xmachine_memory_A ** h_AoS = h_allocate_agent_A_array(a_count);
    for(unsigned int i = 0; i < a_count; i++){
    	h_AoS[i]->id = generate_A_id();
    }

    h_add_agents_A_outputting(h_AoS, a_count);
    h_free_agent_A_array(&h_AoS, a_count);
}


__FLAME_GPU_HOST_FUNC__ void generateBAgents(){
	unsigned int b_count = *get_B_POP();
	if( b_count > xmachine_memory_B_MAX){
		fprintf(stderr, "Error: %u B agents requested. Only room for %u\n", b_count, xmachine_memory_B_MAX);
		fflush(stderr);
		exit(EXIT_FAILURE);
	}

    xmachine_memory_B ** h_AoS = h_allocate_agent_B_array(b_count);
    for(unsigned int i = 0; i < b_count; i++){
    	h_AoS[i]->id = generate_B_id();
    	h_AoS[i]->count = 0;
    }

    h_add_agents_B_inputting(h_AoS, b_count);
    h_free_agent_B_array(&h_AoS, b_count);
}

__FLAME_GPU_INIT_FUNC__ void generatePopulations(){

	// Generate agent(s) of each type.
	generateAAgents();
	generateBAgents();

}

__FLAME_GPU_EXIT_FUNC__ void exitFunction(){

}


__FLAME_GPU_FUNC__ int output(xmachine_memory_A* agent, xmachine_message_msg_list* msg_messages){
    
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    printf("tid %u. A %u outputting message\n", tid, agent->id);

    add_msg_message(msg_messages, agent->id, tid);
    
    return 0;
}

__FLAME_GPU_FUNC__ int input(xmachine_memory_B* agent, xmachine_message_msg_list* msg_messages){
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int count = 0;
    xmachine_message_msg* current_message = get_first_msg_message(msg_messages);
    while (current_message)
    {
        count += 1;
        	
        if (tid < 16) {
        	printf("Tid %u: B %u received from %u, %u \n", tid, agent->id, current_message->id, current_message->threadId);
        }

        current_message = get_next_msg_message(current_message, msg_messages);
    }

    agent->count = count;

    if (tid < 16) {
    	printf("idd %u: B %u received %u messages \n", tid, agent->id, agent->count);
    }
    
    
    return 0;
}

  
#endif // #ifndef _FUNCTIONS_H_
