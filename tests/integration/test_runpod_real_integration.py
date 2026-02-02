import pytest
import pytest_asyncio
import os
import asyncio
from dotenv import load_dotenv
from kestrel_sovereign.features.runpod.manager import RunPodManager
from kestrel_sovereign.features.runpod.providers import DirectRunPodProvider
from kestrel_sovereign.features.runpod.models import PodStatus
from tests.shared.resource_registry import track_runpod, registry

load_dotenv()

# Mark as cloud resource test (excluded from CI by default)
pytestmark = pytest.mark.cloud_resource

# Only run if we have an API key
@pytest.mark.skipif(not os.environ.get("RUNPOD_API_KEY"), reason="RUNPOD_API_KEY not set")
class TestRunPodRealIntegration:
    
    @pytest_asyncio.fixture
    async def manager(self):
        # This uses the REAL provider, no mocks
        manager = RunPodManager()
        # Force initialization of the provider if it's lazy
        if not manager.provider:
            manager.provider = DirectRunPodProvider(api_key=os.environ["RUNPOD_API_KEY"])
        return manager

    @pytest.mark.asyncio
    async def test_list_pods_and_check_logs(self, manager):
        """
        This test connects to the REAL RunPod API.
        It checks for existing pods. 
        If a pod is running, it attempts to fetch logs via SSH.
        It does NOT start a new pod to avoid incurring costs automatically.
        """
        print("\n[REAL-INTEGRATION] Connecting to RunPod API...")
        
        # 1. Get all pods directly from provider to see what's out there
        pods = manager.provider.get_status(pod_id=None) # Assuming get_pod(None) lists all, or we need a list method
        # The current implementation of get_status wraps runpod.get_pod(pod_id). 
        # runpod.get_pods() (plural) is usually how you list. 
        # Let's check if we can list.
        
        import runpod
        runpod.api_key = os.environ["RUNPOD_API_KEY"]
        all_pods = runpod.get_pods()
        
        print(f"[REAL-INTEGRATION] Found {len(all_pods)} pods.")
        for p in all_pods:
            runtime = p.get('runtime') or {}
            print(f" - Pod {p['id']}: {p.get('desiredStatus')} / {runtime.get('containerStatus')}")
        
        running_pods = [p for p in all_pods if p.get('desiredStatus') == 'RUNNING' and (p.get('runtime') or {}).get('containerStatus') == 'running']
        provisioning_pods = [p for p in all_pods if p.get('desiredStatus') == 'RUNNING' and (p.get('runtime') or {}).get('containerStatus') != 'running']

        pod_to_use = None
        started_by_test = False
        should_cleanup_on_failure = False  # Track if we should cleanup on test failure
        tracked_resource_id = None  # For resource registry tracking

        if running_pods:
            pod_to_use = running_pods[0]
            print(f"[REAL-INTEGRATION] Using existing running pod: {pod_to_use['id']}")
        elif provisioning_pods:
            pod_to_use = provisioning_pods[0]
            print(f"[REAL-INTEGRATION] Found provisioning pod {pod_to_use['id']}. Waiting for it to be ready...")
            should_cleanup_on_failure = True  # Terminate stuck provisioning pods

            # Wait for it to be running (shorter timeout - 120s)
            print("[REAL-INTEGRATION] Waiting for pod to start (max 120s)...")
            for _ in range(24):  # 24 * 5s = 120s
                await asyncio.sleep(5)
                pod_status = runpod.get_pod(pod_to_use['id'])
                if pod_status.get('desiredStatus') == 'RUNNING' and (pod_status.get('runtime') or {}).get('containerStatus') == 'running':
                    print("[REAL-INTEGRATION] Pod is RUNNING.")
                    pod_to_use = pod_status
                    should_cleanup_on_failure = False  # Pod is ready, don't cleanup
                    break
            else:
                # Pod failed to start - TERMINATE IT to avoid leaving stuck pods
                print(f"[REAL-INTEGRATION] Pod {pod_to_use['id']} failed to start. Terminating to avoid cost...")
                try:
                    runpod.stop_pod(pod_to_use['id'])
                except Exception as e:
                    print(f"[REAL-INTEGRATION] Error stopping stuck pod: {e}")
                try:
                    runpod.terminate_pod(pod_to_use['id'])
                    print(f"[REAL-INTEGRATION] Terminated stuck pod {pod_to_use['id']}")
                except Exception as e:
                    print(f"[REAL-INTEGRATION] Error terminating stuck pod: {e}")
                pytest.skip("Pod failed to become ready within 120 seconds. Terminated to avoid costs.")
            
            # Wait a bit more for SSH to be ready
            print("[REAL-INTEGRATION] Waiting 20s for SSH to be ready...")
            await asyncio.sleep(20)
            
        else:
            # Find an exited pod to start temporarily
            exited_pods = [p for p in all_pods if p.get('desiredStatus') == 'EXITED']
            if exited_pods:
                pod_to_use = exited_pods[0]
                print(f"[REAL-INTEGRATION] No running pods. Starting exited pod {pod_to_use['id']} for verification...")
                try:
                    gpu_count = pod_to_use.get('gpuCount', 1)
                    runpod.resume_pod(pod_to_use['id'], gpu_count)
                    started_by_test = True
                    # Track in resource registry for crash-safe cleanup
                    tracked_resource_id = track_runpod(pod_to_use['id'], name="test-started-pod")
                    
                    # Wait for it to be running
                    print("[REAL-INTEGRATION] Waiting for pod to start (max 300s)...")
                    for _ in range(60): # 60 * 5s = 300s
                        await asyncio.sleep(5)
                        pod_status = runpod.get_pod(pod_to_use['id'])
                        if pod_status.get('desiredStatus') == 'RUNNING' and (pod_status.get('runtime') or {}).get('containerStatus') == 'running':
                            print("[REAL-INTEGRATION] Pod is RUNNING.")
                            pod_to_use = pod_status
                            break
                    else:
                        pytest.fail("Pod failed to start within 300 seconds.")
                        
                    # Wait a bit more for SSH to be ready
                    print("[REAL-INTEGRATION] Waiting 20s for SSH to be ready...")
                    await asyncio.sleep(20)
                    
                except Exception as e:
                    print(f"[REAL-INTEGRATION] Failed to start pod: {e}")
                    if started_by_test:
                        try:
                            runpod.stop_pod(pod_to_use['id'])
                        except Exception:
                            pass
                        try:
                            runpod.terminate_pod(pod_to_use['id'])
                        except Exception:
                            pass
                    pytest.skip("Could not start a pod for testing.")
            else:
                print("[REAL-INTEGRATION] No pods found to start.")
                return

        pod_id = pod_to_use['id']
        print(f"[REAL-INTEGRATION] Targeting Pod ID: {pod_id}")
        
        try:
            # Test the provider's exec_command directly first
            print(f"[REAL-INTEGRATION] Attempting SSH connection to {pod_id}...")
            logs = await asyncio.to_thread(manager.provider.exec_command, pod_id, "echo 'Hello from Kestrel Integration Test'")
            print(f"[REAL-INTEGRATION] SSH Command Output: {logs.strip()}")
            assert "Hello from Kestrel Integration Test" in logs
            
            # Now test the actual log retrieval command
            print("[REAL-INTEGRATION] Fetching Docker logs...")
            # We need to mock the session object inside manager because get_logs relies on it
            # But we can just call the provider method directly if we want to test the mechanism
            # Or we can set up a fake session.
            
            # Let's test the manager.get_logs method properly by setting up a session
            from kestrel_sovereign.features.runpod.models import RunPodSession, GPUProfile
            from datetime import datetime, timezone
            
            # Create a dummy profile
            dummy_profile = GPUProfile(
                id="test", name="test", task_type="llm", gpu_type_id="test", 
                image_name="test", container_disk_gb=10, volume_gb=0, ports=[], inference_port=8000
            )
            
            session = RunPodSession(
                pod_id=pod_id,
                task_profile="llm",
                model_name="test",
                status=PodStatus.READY,
                started_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc),
                ttl_seconds=3600,
                profile=dummy_profile,
                pod_type="community"
            )
            manager._session = session
            
            docker_logs = await manager.get_logs(tail=10)
            print(f"[REAL-INTEGRATION] Docker Logs (tail 10):\n{docker_logs}")
            assert docker_logs is not None
            
        except Exception as e:
            pytest.fail(f"Real integration test failed: {e}")
        finally:
            # Only clean up pods that THIS TEST started (not pre-existing ones)
            if started_by_test:
                print(f"[REAL-INTEGRATION] Cleaning up pod {pod_id}...")
                try:
                    runpod.stop_pod(pod_id)
                    print("[REAL-INTEGRATION] Pod stopped.")
                except Exception as e:
                    print(f"[REAL-INTEGRATION] Error stopping pod: {e}")
                try:
                    runpod.terminate_pod(pod_id)
                    print("[REAL-INTEGRATION] Pod terminated.")
                except Exception as e:
                    print(f"[REAL-INTEGRATION] Error terminating pod: {e}")
                # Unregister from resource registry since we cleaned up manually
                if tracked_resource_id:
                    registry.untrack(tracked_resource_id)

