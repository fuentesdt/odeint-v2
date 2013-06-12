/*
 * Solves many relaxation equations dxdt = - a * x in parallel and for different values of a.
 * The relaxation equations are completely uncoupled.
 */

#include <stdio.h>
#include <thrust/device_vector.h>

#include <boost/ref.hpp>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/thrust/thrust.hpp>


using namespace std;
using namespace boost::numeric::odeint;

// change to float if your GPU does not support doubles
//typedef double value_type;
typedef float value_type;
typedef thrust::device_vector< value_type > state_type;
typedef runge_kutta4< state_type , value_type , state_type , value_type > stepper_type;

struct mean_field_calculator
{
    struct sin_functor : public thrust::unary_function< value_type , value_type >
    {
        __host__ __device__
        value_type operator()( value_type x) const
        {
            //return sin( x );
            return x ;
        }
    };

    struct cos_functor : public thrust::unary_function< value_type , value_type >
    {
        __host__ __device__
        value_type operator()( value_type x) const
        {
            //return cos( x );
            return x ;
        }
    };

    static std::pair< value_type , value_type > get_mean( const state_type &x )
    {
        //[ thrust_phase_ensemble_sin_sum
        value_type sin_sum = thrust::reduce(
                thrust::make_transform_iterator( x.begin() , sin_functor() ) ,
                thrust::make_transform_iterator( x.end() , sin_functor() ) );
        //]
        value_type cos_sum = thrust::reduce(
                thrust::make_transform_iterator( x.begin() , cos_functor() ) ,
                thrust::make_transform_iterator( x.end() , cos_functor() ) );

        cos_sum /= value_type( x.size() );
        sin_sum /= value_type( x.size() );

        value_type K = sqrt( cos_sum * cos_sum + sin_sum * sin_sum );
        value_type Theta = atan2( sin_sum , cos_sum );

        return std::make_pair( K , Theta );
    }
};
// FIXME need FFT observer
struct statistics_observer
{
    value_type m_K_mean;
    size_t m_count;

    statistics_observer( void )
    : m_K_mean( 0.0 ) , m_count( 0 ) { }

    template< class State >
    void operator()( const State &x , value_type t )
    {
        std::pair< value_type , value_type > mean = mean_field_calculator::get_mean( x );
        m_K_mean += mean.first;
        ++m_count;
    }

    value_type get_K_mean( void ) const { return ( m_count != 0 ) ? m_K_mean / value_type( m_count ) : 0.0 ; }

    void reset( void ) { m_K_mean = 0.0; m_count = 0; }
};
struct relaxation
{
    struct relaxation_functor
    {
        template< class T >
        __host__ __device__
        void operator()( T t ) const
        {
            // unpack the parameter we want to vary and the Lorenz variables
            value_type pyr = thrust::get< 0 >( t );
            value_type lac = thrust::get< 1 >( t );
            value_type a = thrust::get< 2 >( t );
            value_type b = thrust::get< 3 >( t );
            value_type c = thrust::get< 4 >( t );
            thrust::get< 5 >( t ) = b*lac - a * pyr + c;
            thrust::get< 6 >( t ) = c*pyr - a * lac + b;
        }
    };

    relaxation( size_t N , const state_type &a, const state_type &b, const state_type &c )
    : m_N( N ) , m_a( a ) , m_b( b ), m_c( c ){ }

    void operator()(  const state_type &x , state_type &dxdt , value_type t ) const
    {
        thrust::for_each(
            thrust::make_zip_iterator( thrust::make_tuple( 
                                       x.begin() , 
                                       x.begin() + m_N ,
                                       m_a.begin() , 
                                       m_b.begin() , 
                                       m_c.begin() , 
                                       dxdt.begin() ,
                                       dxdt.begin() + m_N   
                                      ) ) ,
            thrust::make_zip_iterator( thrust::make_tuple( 
                                       x.begin() + m_N ,
                                       x.end() , 
                                       m_a.end() ,
                                       m_b.end() ,
                                       m_c.end() ,
                                       dxdt.begin() + m_N  , 
                                       dxdt.end() 
                                      ) ) ,
            relaxation_functor() );
    }

    size_t m_N;
    const state_type &m_a;
    const state_type &m_b;
    const state_type &m_c;
};

int main( int argc , char* argv[] )
{
    if( argc < 2 )
      {
      std::cerr << "Missing Parameters " << std::endl;
      std::cerr << "Usage: " << argv[0];
      std::cerr << " GPUID [NSIZE] [FINALTIME] [DELTAT] [STARTTIME] ";
      std::cerr <<  std::endl;
      return EXIT_FAILURE;
      }
    enum {  GPUID = 1, NSIZE , FINALTIME, DELTAT, STARTTIME };

    int deviceNum= atoi( argv[GPUID] );

    value_type dt = 0.01;
    if( argc > DELTAT )
      {
      dt = atof( argv[DELTAT] );
      }
    size_t Nsize = 1024*2;
    if( argc > NSIZE )
      {
      Nsize = atoi( argv[NSIZE] );
      }
    value_type final_time = 10.0;
    if( argc > FINALTIME )
      {
      final_time = atof( argv[FINALTIME] );
      }
    value_type start_time = 0.0;
    if( argc > STARTTIME )
      {
      start_time = atof( argv[STARTTIME] );
      }

    cudaError      ierrCuda;
    ierrCuda =  cudaSetDevice(deviceNum);
    cout <<  " reseting GPU: " << endl; 
    ierrCuda = cudaDeviceReset();
    int driver_version , runtime_version;
    cudaDriverGetVersion( &driver_version );
    cudaRuntimeGetVersion ( &runtime_version );
    cout << driver_version << "\t" << runtime_version << endl;

    cudaDeviceProp deviceProp;
    if (cudaGetDeviceProperties(&deviceProp, deviceNum) == cudaSuccess) {
      printf( " Device %d: %s %d.%d\n", deviceNum, deviceProp.name,deviceProp.major,deviceProp.minor);
      printf(" Global memory available on device in bytes %d\n"                            ,  deviceProp.totalGlobalMem                  );
      printf(" Shared memory available per block in bytes %d\n"                            ,  deviceProp.sharedMemPerBlock               );
      printf(" 32-bit registers available per block %d\n"                                  ,  deviceProp.regsPerBlock                    );
      printf(" Warp size in threads %d\n"                                                  ,  deviceProp.warpSize                        );
      printf(" Maximum pitch in bytes allowed by memory copies %d\n"                       ,  deviceProp.memPitch                        );
      printf(" Maximum number of threads per block %d\n"                                   ,  deviceProp.maxThreadsPerBlock              );
      printf(" Maximum size of each dimension of a block %d\n"                             ,  deviceProp.maxThreadsDim[0]                );
      printf(" Maximum size of each dimension of a block %d\n"                             ,  deviceProp.maxThreadsDim[1]                );
      printf(" Maximum size of each dimension of a block %d\n"                             ,  deviceProp.maxThreadsDim[2]                );
      printf(" Maximum size of each dimension of a grid %d\n"                              ,  deviceProp.maxGridSize[0]                  );
      printf(" Maximum size of each dimension of a grid %d\n"                              ,  deviceProp.maxGridSize[1]                  );
      printf(" Maximum size of each dimension of a grid %d\n"                              ,  deviceProp.maxGridSize[2]                  );
      printf(" Clock frequency in kilohertz %d\n"                                          ,  deviceProp.clockRate                       );
      printf(" Constant memory available on device in bytes %d\n"                          ,  deviceProp.totalConstMem                   );
      printf(" Alignment requirement for textures %d\n"                                    ,  deviceProp.textureAlignment                );
      printf(" Number of multiprocessors on device %d\n"                                   ,  deviceProp.multiProcessorCount             );
      printf(" Specified whether there is a run time limit on kernels %d\n"                ,  deviceProp.kernelExecTimeoutEnabled        );
      printf(" Device is integrated as opposed to discrete %d\n"                           ,  deviceProp.integrated                      );
      printf(" Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer %d\n",  deviceProp.canMapHostMemory                );
      printf(" Compute mode (See ::cudaComputeMode) %d\n"                                  ,  deviceProp.computeMode                     );
      printf(" Maximum 1D texture size %d\n"                                               ,  deviceProp.maxTexture1D                    );
      printf(" Maximum 2D texture dimensions %d\n"                                         ,  deviceProp.maxTexture2D[0]                 );
      printf(" Maximum 2D texture dimensions %d\n"                                         ,  deviceProp.maxTexture2D[1]                 );
      printf(" Maximum 3D texture dimensions %d\n"                                         ,  deviceProp.maxTexture3D[0]                 );
      printf(" Maximum 3D texture dimensions %d\n"                                         ,  deviceProp.maxTexture3D[1]                 );
      printf(" Maximum 3D texture dimensions %d\n"                                         ,  deviceProp.maxTexture3D[2]                 );
      printf(" Maximum 1D layered texture dimensions %d\n"                                 ,  deviceProp.maxTexture1DLayered[0]          );
      printf(" Maximum 1D layered texture dimensions %d\n"                                 ,  deviceProp.maxTexture1DLayered[1]          );
      printf(" Maximum 2D layered texture dimensions %d\n"                                 ,  deviceProp.maxTexture2DLayered[0]          );
      printf(" Maximum 2D layered texture dimensions %d\n"                                 ,  deviceProp.maxTexture2DLayered[1]          );
      printf(" Maximum 2D layered texture dimensions %d\n"                                 ,  deviceProp.maxTexture2DLayered[2]          );
      printf(" Alignment requirements for surfaces %d\n"                                   ,  deviceProp.surfaceAlignment                );
      printf(" Device can possibly execute multiple kernels concurrently %d\n"             ,  deviceProp.concurrentKernels               );
      printf(" Device has ECC support enabled %d\n"                                        ,  deviceProp.ECCEnabled                      );
      printf(" PCI bus ID of the device %d\n"                                              ,  deviceProp.pciBusID                        );
      printf(" PCI device ID of the device %d\n"                                           ,  deviceProp.pciDeviceID                     );
      printf(" PCI domain ID of the device %d\n"                                           ,  deviceProp.pciDomainID                     );
      printf(" 1 if device is a Tesla device using TCC driver, 0 otherwise %d\n"           ,  deviceProp.tccDriver                       );
      printf(" Number of asynchronous engines %d\n"                                        ,  deviceProp.asyncEngineCount                );
      printf(" Device shares a unified address space with the host %d\n"                   ,  deviceProp.unifiedAddressing               );
      printf(" Peak memory clock frequency in kilohertz %d\n"                              ,  deviceProp.memoryClockRate                 );
      printf(" Global memory bus width in bits %d\n"                                       ,  deviceProp.memoryBusWidth                  );
      printf(" Size of L2 cache in bytes %d\n"                                             ,  deviceProp.l2CacheSize                     );
      printf(" Maximum resident threads per multiprocessor %d\n"                           ,  deviceProp.maxThreadsPerMultiProcessor     );
    } else {
      printf(" Unable to determine device %d properties, exiting\n",deviceNum);
      return 1;
    }

    //[ thrust_lorenz_parameters_define_beta
    vector< value_type > alpha_host( Nsize );
    vector< value_type >  beta_host( Nsize );
    vector< value_type > gamma_host( Nsize );
    // FIXME read from CMD line
    const value_type alpha_min = 0.0 , alpha_max = 56.0;
    const value_type  beta_min = 0.0 ,  beta_max = 26.0;
    const value_type gamma_min = 0.0 , gamma_max = 86.0;
    for( size_t i=0 ; i<Nsize ; ++i )
       {
        alpha_host[i] = alpha_min + value_type( i ) * ( alpha_max - alpha_min ) / value_type( Nsize - 1 );
         beta_host[i] =  beta_min + value_type( i ) * (  beta_max -  beta_min ) / value_type( Nsize - 1 );
        gamma_host[i] = gamma_min + value_type( i ) * ( gamma_max - gamma_min ) / value_type( Nsize - 1 );
       }

    cout << " copy vectors " << endl;
    state_type alpha = alpha_host;
    state_type  beta =  beta_host;
    state_type gamma = gamma_host;

    // initialize the intial state x
    state_type x( 2*Nsize );
    thrust::fill( x.begin()              , x.begin() + 1 * Nsize , 10.0 );
    thrust::fill( x.begin() + 1 * Nsize  , x.end()               , 1.0 );

    // initialize observer
    statistics_observer obs;

    // integrate
    cout << " integrate " << endl;
    relaxation relax( Nsize , alpha, beta, gamma );
    integrate_const( stepper_type() , boost::ref( relax ) , x , start_time , final_time , dt, boost::ref( obs ) );
    cout << "count " << obs.m_count << " mean \t" << obs.get_K_mean() << endl;
    
    // write final state
    //thrust::copy( x.begin() , x.end() , std::ostream_iterator< value_type >( std::cout , "\n" ) );
    //std::cout << std::endl;
    return 0;
}
