
__device__ float4 wav2color(float wav)
{
    int w ; 
    if (isinf(wav))      w = -1 ;
    else if( isnan(wav)) w = -2 ;  
    else                 w = int(wav) ;

    float3 col ;
 
    if      (w >= 380 && w < 440)  col = (float3) ( -(wav - 440.) / (440. - 350.) , 0.0, 1.0 ) ;
    else if (w >= 440 && w < 490)  col = (float3) ( 0.0, (wav - 440.) / (490. - 440.), 1.0 ) ;
    else if (w >= 490 && w < 510)  col = (float3) ( 0.0, 1.0, -(wav - 510.) / (510. - 490.) ) ;
    else if (w >= 510 && w < 580)  col = (float3) ( (wav - 510.) / (580. - 510.), 1.0 , 0.0 ) ;
    else if (w >= 580 && w < 645)  col = (float3) ( 1.0,  -(wav - 645.) / (645. - 580.), 0.0 );
    else if (w >= 645 && w <= 780) col = (float3) ( 1.0, 0.0, 0.0 ) ;
    else if (w < 0)                col = (float3) ( 1.0, 1.0, 1.0 ) ;
    else                           col = (float3) ( 0.0, 0.0, 0.0 );

    // intensity correction
    float SSS ;
    if     ( w >= 380 && w < 420 ) SSS = 0.3 + 0.7*(wav - 350.) / (420. - 350.);
    else if (w >= 420 && w <= 700) SSS = 1.0 ;
    else if (w > 700 && w <= 780)  SSS = 0.3 + 0.7*(780. - wav) / (780. - 700.);
    else SSS = 1.0 ;  // formerly 0, but want to see the weird ones


    return make_float4( SSS*col.x, SSS*col.y, SSS*col.z, 1.) ; 

}



