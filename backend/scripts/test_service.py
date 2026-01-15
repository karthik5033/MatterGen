import sys
import os
import asyncio
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from app.services.material_service import material_service
    print("Service imported")
    
    async def test():
        print("Generating...")
        res = await material_service.generate_candidates("battery", {}, 1)
        print("Result:", res)

    asyncio.run(test())
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
