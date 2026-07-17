`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/16/2026 11:21:01 AM
// Design Name: 
// Module Name: lif
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


module lif #(

    parameter BITWIDTH = 16,
    parameter N_INPUTS = 784,
    parameter THRESHOLD = 1
    
)(
    input wire rst,
    input wire clk,
    input wire [N_INPUTS-1:0] spike_in,
    input wire signed [BITWIDTH-1:0] weights [0:N_INPUTS-1], // size of each bit and no of weights respectively

    output reg spike_out
);

    reg signed[BITWIDTH-1:0] membrane;
    reg signed[BITWIDTH-1:0] after_integrate;
    wire signed[BITWIDTH-1:0] after_leak;
    integer i; 
    
    assign after_leak = membrane - (membrane>>>1); // 50% remains after leak 
    
    always @(*) begin 
        
        after_integrate = after_leak;
        
        for(i = 0;i<N_INPUTS;i=i+1)begin 
            if(spike_in[i])
                after_integrate = after_integrate + weights[i];
                
        end 
        
    end 
    
    always @(posedge clk or posedge rst) begin 
    
        if(rst) begin
            membrane <= 0;
            spike_out <= 0;
            
        end 
        
        else if(after_integrate>=THRESHOLD) begin 
        
            membrane <= 0;   //reset , hard reset
            spike_out <= 1;  //fire
            
        end 
        
        else begin 
        
            membrane <= after_integrate;
            spike_out <= 0; 
            
        end 
        
        
        
    end 
    

endmodule
