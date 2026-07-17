`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 07/08/2026 02:55:08 PM
// Design Name: 
// Module Name: snn_layer
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module snn_layer #( // hidden layer

    parameter NUM_NEURONS = 128,//hidden layer
    parameter NUM_INPUTS = 784,  // pixels
    parameter BITWIDTH = 16,
    parameter THRESHOLD = 1,
    parameter MEM_FILE = "W1.mem"

)(

    input wire clk ,
    input wire rst , 
    input wire [NUM_INPUTS-1:0] spike_in,  // input spike
    output wire [NUM_NEURONS-1:0] spike_out // output spike
    
);

    // the weights are fetched from the bram 
    
    // weight array : weights[neuron][input]
    
    wire signed [BITWIDTH-1:0] weights[0:NUM_NEURONS-1][0:NUM_INPUTS-1];
    
    
    // instantiate one weight_bram per neuron 
    
    genvar n,i; 
    
    generate 
    
        for(n = 0;n<NUM_NEURONS;n=n+1) begin : neuron_bram 
        
            //small bram for this neuron weights
            reg signed [BITWIDTH-1:0] w_mem [0:NUM_INPUTS-1]; 
            
            initial begin 
            
                //each neuron reads weights from .mem file
                $readmemh(MEM_FILE,w_mem,n*NUM_INPUTS,(n+1)*NUM_INPUTS-1); // 3rd and 4th arguments are start and end addresses
                
            end 
            
            
            // wire wrights out 
            
            for(i = 0;i<NUM_INPUTS;i=i+1) begin : wire_weights 
            
                assign weights[n][i] = w_mem[i];
                
            end 
            
            
         end
                   
    endgenerate 
    
    // instantiate each lif neuron 
    
    generate 
    
        for(n = 0;n<NUM_NEURONS;n=n+1) begin : neuron_array 
        
        
            lif # (
            
                .BITWIDTH(BITWIDTH),
                .N_INPUTS(NUM_INPUTS),
                .THRESHOLD(THRESHOLD)
                
            ) lif_inst (
            
                .clk(clk),
                .rst(rst),
                .spike_in(spike_in),
                .weights(weights[n]),
                .spike_out(spike_out[n])
                
            );
            
            
            
        end
        
        
   endgenerate 
    
    
            

endmodule
